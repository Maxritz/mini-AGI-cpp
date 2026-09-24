// src/paged.cpp
// Paged pool for the deterministic inference reference: Tiers (an LRU
// disk/RAM cache over e%05d.npz files, section 2) and PagedPool (the surface
// PooledMLP routes into, section 3). Training-only pieces -- growth, pruning,
// Adam-moment write-back, aux loss -- are excluded per the TIER-2/EXCLUDED
// note; Tier-2 keys/demand/telemetry round-trip are kept for the
// self-consistency test.
#include "paged.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace minagi::paged {

namespace {

bool ends_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::string strip_npy(const std::string& name) {
    if (ends_with(name, ".npy") && name.size() > 4) {
        return name.substr(0, name.size() - 4);
    }
    return name;
}

std::string expert_file_name(int uid) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "e%05d.npz", uid);
    return std::string(buf);
}

mt::Tensor zeros(const mt::Shape& s) {
    return mt::make_zeros(s, mt::DType::FP32);
}

}  // namespace

mt::Tensor array_to_tensor(const mininpz::Array& arr) {
    if (arr.dtype != mininpz::DType::F4 || arr.shape.rank == 0 ||
        arr.shape.rank > 4) {
        mt::Shape s;
        s.rank = 0;
        return mt::make_zeros(s, mt::DType::FP32);
    }
    mt::Tensor out;
    out.dtype = mt::DType::FP32;
    out.shape.rank = arr.shape.rank;
    for (size_t i = 0; i < arr.shape.rank; ++i) {
        out.shape.d[i] = static_cast<int64_t>(arr.shape.d[i]);
    }
    out.data_ = arr.bytes;
    return out;
}

mininpz::Array tensor_to_array(const mt::Tensor& t) {
    mininpz::Array out;
    out.dtype = mininpz::DType::F4;
    out.shape.rank = t.shape.rank;
    for (size_t i = 0; i < t.shape.rank && i < 4; ++i) {
        out.shape.d[i] = static_cast<size_t>(t.shape.d[i]);
    }
    if (t.dtype == mt::DType::FP32) {
        out.bytes = t.data_;
    } else {
        // BF16 -> fp32 on demand; the reference never writes bf16.
        const size_t n = static_cast<size_t>(t.shape.numel());
        out.bytes.resize(n * 4);
        const uint16_t* src = t.ptr<uint16_t>();
        for (size_t i = 0; i < n; ++i) {
            float f = mt::bfloat16{src[i]}.to_float();
            std::memcpy(out.bytes.data() + i * 4, &f, sizeof(float));
        }
    }
    return out;
}

// ---- Tiers ---------------------------------------------------------------

Tiers::Tiers(const std::string& experts_dir, int d_model, int d_ff,
             int ram_capacity, bool read_only)
    : path_(experts_dir), d_model_(d_model), d_ff_(d_ff),
      ram_capacity_(ram_capacity), read_only_(read_only) {}

std::string Tiers::file_path(int uid) const {
    return (fs::path(path_) / expert_file_name(uid)).string();
}

bool Tiers::from_disk(int uid, Entry& out) {
    std::vector<mininpz::NpzEntry> entries;
    if (!mininpz::read_npz(file_path(uid), entries)) {
        return false;
    }
    bool have_w1 = false, have_w3 = false, have_w2 = false;
    for (const auto& e : entries) {
        const std::string name = strip_npy(e.name);
        mt::Tensor t = array_to_tensor(e.arr);
        if (name == "w1") {
            out.w1 = std::move(t);
            have_w1 = true;
        } else if (name == "w3") {
            out.w3 = std::move(t);
            have_w3 = true;
        } else if (name == "w2") {
            out.w2 = std::move(t);
            have_w2 = true;
        }
        // moment keys (_m/_v) are ignored: inference does not hold Adam state.
    }
    return have_w1 && have_w3 && have_w2;
}

bool Tiers::to_disk(int uid, const Entry& e) {
    if (read_only_) {
        return false;
    }
    std::vector<mininpz::NpzEntry> arrays;
    arrays.push_back({"w1.npy", tensor_to_array(e.w1)});
    arrays.push_back({"w3.npy", tensor_to_array(e.w3)});
    arrays.push_back({"w2.npy", tensor_to_array(e.w2)});
    const std::string target = file_path(uid);
    const std::string tmp = target + ".tmp.npz";
    if (fs::exists(tmp)) {
        fs::remove(tmp);
    }
    if (!mininpz::write_npz(tmp, arrays)) {
        return false;
    }
    if (fs::exists(target)) {
        std::error_code ec;
        fs::remove(target, ec);
    }
    std::error_code ec;
    fs::rename(tmp, target, ec);
    if (ec) {
        fs::remove(tmp);
        return false;
    }
    return true;
}

void Tiers::move_mru(int uid) {
    auto it = std::find(order_.begin(), order_.end(), uid);
    if (it != order_.end()) {
        order_.erase(it);
    }
    order_.push_back(uid);
}

bool Tiers::fetch(int uid, mt::Tensor& w1, mt::Tensor& w3, mt::Tensor& w2) {
    auto found = ram_.find(uid);
    if (found != ram_.end()) {
        counters_.hits++;
        move_mru(uid);
        w1 = found->second.w1;
        w3 = found->second.w3;
        w2 = found->second.w2;
        return true;
    }
    counters_.reads++;
    Entry e;
    if (!from_disk(uid, e)) {
        return false;
    }
    ram_[uid] = e;
    move_mru(uid);
    trim();
    w1 = ram_[uid].w1;
    w3 = ram_[uid].w3;
    w2 = ram_[uid].w2;
    return true;
}

void Tiers::put(int uid, const mt::Tensor& w1, const mt::Tensor& w3,
                const mt::Tensor& w2, bool dirty) {
    Entry e;
    e.w1 = w1;
    e.w3 = w3;
    e.w2 = w2;
    ram_[uid] = std::move(e);
    move_mru(uid);
    if (dirty && !read_only_) {
        if (dirty_.insert(uid).second) {
            dirty_order_.push_back(uid);
        }
    }
    trim();
}

void Tiers::trim() {
    while (ram_.size() > static_cast<size_t>(ram_capacity_)) {
        if (order_.empty()) {
            break;
        }
        const int uid = order_.front();
        order_.pop_front();
        auto it = ram_.find(uid);
        if (it == ram_.end()) {
            continue;
        }
        Entry ent = std::move(it->second);
        ram_.erase(it);
        counters_.evictions++;
        if (dirty_.count(uid)) {
            if (to_disk(uid, ent)) {
                counters_.writebacks++;
            }
            dirty_.erase(uid);
            dirty_order_.erase(std::remove(dirty_order_.begin(), dirty_order_.end(), uid),
                               dirty_order_.end());
        }
    }
}

void Tiers::flush() {
    if (read_only_) {
        return;
    }
    for (int uid : dirty_order_) {
        auto it = ram_.find(uid);
        if (it == ram_.end()) {
            dirty_.erase(uid);
            continue;
        }
        if (to_disk(uid, it->second)) {
            counters_.writebacks++;
        }
        dirty_.erase(uid);
    }
    dirty_order_.clear();
}

const TierCounters& Tiers::counters() const { return counters_; }

void Tiers::reset_counters() {
    counters_.reads = 0;
    counters_.hits = 0;
    counters_.evictions = 0;
    counters_.writebacks = 0;
}

// ---- PagedPool ------------------------------------------------------------

PagedPool::PagedPool(int n_experts, int d_model, int pool_d_ff, int resident,
                     int ram_capacity, double explore, double margin, int dwell,
                     bool read_only, mt::Tensor gate_init)
    : n_experts_(n_experts), d_model_(d_model), d_ff_(pool_d_ff),
      resident_(std::max(0, std::min(resident, n_experts))), explore_(explore),
      margin_(margin), dwell_(dwell), read_only_(read_only),
      ram_capacity_(ram_capacity), slots_(static_cast<size_t>(std::max(0, resident_)), -1),
      next_uid_(n_experts), segments_(0), swaps_(0), have_telemetry_(false),
      experts_dir_set_(false), keys_built_(false) {
    mt::Shape s1;
    s1.rank = 1;
    s1.d[0] = n_experts;
    use_ = zeros(s1);
    age_ = zeros(s1);
    born_ = zeros(s1);
    gate_seen_ = zeros(s1);
    last_seen_ = zeros(s1);
    since_ = zeros(s1);
    admits_ = zeros(s1);

    if (gate_init.dtype == mt::DType::FP32 && gate_init.shape.rank == 1 &&
        gate_init.shape.d[0] == n_experts) {
        gate_ = gate_init;
    } else {
        gate_ = mt::make(s1, mt::DType::FP32, 1.0f);
    }

    mt::Shape s2;
    s2.rank = 2;
    // [resident, d_ff, d]
    mt::Shape s3;
    s3.rank = 3;
    s3.d[0] = resident_;
    s3.d[1] = d_ff_;
    s3.d[2] = d_model_;
    w1_ = zeros(s3);
    w3_ = zeros(s3);
    s3.d[1] = d_model_;
    s3.d[2] = d_ff_;
    w2_ = zeros(s3);

    ever_.assign(static_cast<size_t>(n_experts), 0);
    uid_.resize(static_cast<size_t>(n_experts));
    for (int i = 0; i < n_experts; ++i) {
        uid_[static_cast<size_t>(i)] = i;
    }

    s2.rank = 1;
    s2.d[0] = d_model_;
    summary_ = zeros(s2);

    mt::Shape sseg;
    sseg.rank = 2;
    sseg.d[0] = std::max(n_experts, 1);
    sseg.d[1] = d_model_;
    segment_router_ = zeros(sseg);

    keys_ = zeros(sseg);
}

void PagedPool::set_experts_dir(const std::string& experts_dir) {
    experts_dir_ = experts_dir;
    tiers_ = std::make_unique<Tiers>(experts_dir, d_model_, d_ff_,
                                     ram_capacity_, read_only_);
    experts_dir_set_ = true;
}

bool PagedPool::have_telemetry() const { return have_telemetry_; }

int PagedPool::n_experts() const { return n_experts_; }
int PagedPool::n_routable() const { return resident_; }
int PagedPool::router_rows() const { return n_experts_; }
const std::vector<int>& PagedPool::slots() const { return slots_; }
const mt::Tensor& PagedPool::gate() const { return gate_; }
const mt::Tensor& PagedPool::w1() const { return w1_; }
const mt::Tensor& PagedPool::w3() const { return w3_; }
const mt::Tensor& PagedPool::w2() const { return w2_; }

void PagedPool::note_use(const std::vector<long long>& slot_hits) const {
    float* u = use_.ptr<float>();
    const size_t hit_n = slot_hits.size();
    for (size_t j = 0; j < slots_.size(); ++j) {
        if (j >= hit_n) break;
        const int slot_uid = slots_[j] > 0 ? slots_[j] : 0;
        if (slot_uid >= 0 && slot_uid < n_experts_) {
            u[static_cast<size_t>(slot_uid)] += static_cast<float>(slot_hits[j]);
        }
    }
    float* a = age_.ptr<float>();
    for (int i = 0; i < n_experts_; ++i) {
        a[static_cast<size_t>(i)] += 1.0f;
    }
}

std::vector<int> PagedPool::resident_rows() const {
    std::vector<int> rows;
    rows.reserve(slots_.size());
    for (int s : slots_) {
        rows.push_back(s > 0 ? s : 0);
    }
    return rows;
}

mt::Tensor PagedPool::routable_gate() const {
    mt::Shape s;
    s.rank = 1;
    s.d[0] = static_cast<int64_t>(slots_.size());
    mt::Tensor out = zeros(s);
    float* op = out.ptr<float>();
    const float* g = gate_.ptr<float>();
    for (size_t j = 0; j < slots_.size(); ++j) {
        const int thumb = slots_[j] > 0 ? slots_[j] : 0;
        op[j] = (thumb < n_experts_) ? g[static_cast<size_t>(thumb)] : 0.0f;
    }
    return out;
}

double PagedPool::segment_score(int e) const {
    const float* w = segment_router_.ptr<float>();
    const float* sp = summary_.ptr<float>();
    const int64_t d = summary_.shape.d[0];
    double acc = 0.0;
    for (int64_t dd = 0; dd < d; ++dd) {
        acc += static_cast<double>(w[static_cast<size_t>(e) * static_cast<size_t>(d) + static_cast<size_t>(dd)]) *
               static_cast<double>(sp[static_cast<size_t>(dd)]);
    }
    const float g = gate_.ptr<float>()[static_cast<size_t>(e)];
    acc += 0.5 * std::log1p(static_cast<double>(std::fabs(g)));
    return acc;
}

std::vector<int> PagedPool::choose() {
    const int k = std::min(resident_, n_experts_);
    if (k <= 0) {
        return {};
    }
    // n_explore = min(int(k*explore), max(n_experts-k,0))
    const int n_explore = std::min(static_cast<int>(static_cast<double>(k) * explore_),
                                   std::max(n_experts_ - k, 0));
    const int n_reliable = k - n_explore;

    std::vector<std::pair<double, int>> scored;
    scored.reserve(static_cast<size_t>(n_experts_));
    for (int e = 0; e < n_experts_; ++e) {
        scored.emplace_back(segment_score(e), e);
    }
    std::partial_sort(scored.begin(), scored.begin() + n_reliable, scored.end(),
                      [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                          if (a.first != b.first) return a.first > b.first;
                          return a.second < b.second;
                      });

    std::vector<int> chosen;
    for (int i = 0; i < n_reliable; ++i) {
        chosen.push_back(scored[static_cast<size_t>(i)].second);
    }
    if (n_explore > 0) {
        std::vector<double> stale;
        stale.reserve(static_cast<size_t>(n_experts_));
        const float* ls = last_seen_.ptr<float>();
        for (int e = 0; e < n_experts_; ++e) {
            stale.push_back(std::find(chosen.begin(), chosen.end(), e) != chosen.end()
                                ? std::numeric_limits<double>::infinity()
                                : static_cast<double>(ls[static_cast<size_t>(e)]));
        }
        std::vector<std::pair<double, int>> order;
        for (int e = 0; e < n_experts_; ++e) {
            order.emplace_back(stale[static_cast<size_t>(e)], e);
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                             if (a.first != b.first) return a.first < b.first;
                             return a.second < b.second;
                         });
        for (int i = 0; i < n_explore && i < static_cast<int>(order.size()); ++i) {
            chosen.push_back(order[static_cast<size_t>(i)].second);
        }
    }
    if (chosen.size() > static_cast<size_t>(k)) {
        chosen.resize(static_cast<size_t>(k));
    }
    return chosen;
}

void PagedPool::refresh_keys() {
    for (size_t slot = 0; slot < slots_.size(); ++slot) {
        const int e = slots_[slot];
        if (e < 0 || e >= n_experts_) {
            continue;
        }
        // slice w1 slot row -> [d_ff, d]
        mt::Shape s;
        s.rank = 2;
        s.d[0] = w1_.shape.d[1];
        s.d[1] = w1_.shape.d[2];
        mt::Tensor row = zeros(s);
        const float* src = w1_.ptr<float>() + slot * static_cast<size_t>(w1_.shape.d[1]) * static_cast<size_t>(w1_.shape.d[2]);
        std::memcpy(row.ptr<float>(), src, static_cast<size_t>(w1_.shape.d[1]) * static_cast<size_t>(w1_.shape.d[2]) * sizeof(float));
        mt::Tensor key = key_of(row);
        float* kp = keys_.ptr<float>();
        std::memcpy(kp + static_cast<size_t>(e) * static_cast<size_t>(d_model_), key.ptr<float>(),
                    static_cast<size_t>(d_model_) * sizeof(float));
    }
}

int PagedPool::swap_to(const std::vector<int>& suggested) {
    // de-dup preserving order, slice to resident, pad with the lowest missing.
    std::vector<int> ids;
    for (int i : suggested) {
        if (std::find(ids.begin(), ids.end(), i) == ids.end()) {
            ids.push_back(i);
        }
    }
    if (static_cast<int>(ids.size()) > resident_) {
        ids.resize(static_cast<size_t>(resident_));
    }
    for (int cand = 0; static_cast<int>(ids.size()) < resident_ && cand < n_experts_; ++cand) {
        if (std::find(ids.begin(), ids.end(), cand) == ids.end()) {
            ids.push_back(cand);
        }
    }

    segments_ += 1;
    float* since = since_.ptr<float>();
    float* admits = admits_.ptr<float>();
    float* last_seen = last_seen_.ptr<float>();
    std::vector<char>* ever = &ever_;
    for (int i : ids) {
        if (i >= 0 && i < n_experts_ &&
            std::find(slots_.begin(), slots_.end(), i) == slots_.end()) {
            since[static_cast<size_t>(i)] = static_cast<float>(segments_);
            admits[static_cast<size_t>(i)] += 1.0f;
        }
    }
    if (segments_ % 8 == 0) {
        refresh_keys();
    }
    for (int i : ids) {
        if (i >= 0 && i < n_experts_) {
            last_seen[static_cast<size_t>(i)] = static_cast<float>(segments_);
            (*ever)[static_cast<size_t>(i)] = 1;
        }
    }

    std::map<int, int> here;  // uid -> slot for occupied slots
    for (size_t s = 0; s < slots_.size(); ++s) {
        if (slots_[s] >= 0) {
            here[slots_[s]] = static_cast<int>(s);
        }
    }
    std::vector<int> plan(static_cast<size_t>(resident_), -1);
    std::vector<char> kept(static_cast<size_t>(resident_), 0);
    for (int e : ids) {
        auto it = here.find(e);
        if (it != here.end()) {
            plan[static_cast<size_t>(it->second)] = e;
            kept[static_cast<size_t>(it->second)] = 1;
        }
    }
    std::vector<int> free_slots;
    for (int s = 0; s < resident_; ++s) {
        if (!kept[static_cast<size_t>(s)]) {
            free_slots.push_back(s);
        }
    }
    for (int e : ids) {
        if (here.count(e) == 0) {
            if (free_slots.empty()) {
                break;
            }
            plan[static_cast<size_t>(free_slots.front())] = e;
            free_slots.erase(free_slots.begin());
        }
    }
    if (plan == slots_) {
        return 0;
    }
    int loads = 0;
    for (int e : plan) {
        if (e >= 0 && here.count(e) == 0) {
            ++loads;
        }
    }

    // _rearrange: park what leaves, fetch what arrives.
    if (tiers_) {
        const std::vector<int> old = slots_;
        const int64_t w1s = w1_.shape.d[1] * w1_.shape.d[2];
        const int64_t w3s = w3_.shape.d[1] * w3_.shape.d[2];
        const int64_t w2s = w2_.shape.d[1] * w2_.shape.d[2];
        for (int e : old) {
            if (e < 0 || std::find(plan.begin(), plan.end(), e) != plan.end()) {
                continue;
            }
            auto hi = here.find(e);
            if (hi == here.end()) {
                continue;
            }
            const int s = hi->second;
            mt::Tensor p1 = zeros(mt::Shape{});
            mt::Tensor p3 = zeros(mt::Shape{});
            mt::Tensor p2 = zeros(mt::Shape{});
            row_of(w1_, s, w1s, p1);
            row_of(w3_, s, w3s, p3);
            row_of(w2_, s, w2s, p2);
            tiers_->put(static_cast<int>(uid_[static_cast<size_t>(e)]), p1, p3, p2,
                        true);
        }
        for (size_t s = 0; s < plan.size(); ++s) {
            const int e = plan[s];
            if (e < 0 || old[static_cast<size_t>(s)] == e) {
                continue;
            }
            mt::Tensor f1, f3, f2;
            if (tiers_->fetch(static_cast<int>(uid_[static_cast<size_t>(e)]), f1, f3,
                              f2)) {
                set_row(w1_, static_cast<int>(s), f1);
                set_row(w3_, static_cast<int>(s), f3);
                set_row(w2_, static_cast<int>(s), f2);
            }
        }
    }
    slots_ = plan;
    swaps_ += 1;
    return loads;
}

void PagedPool::observe(const mt::Tensor& segment_mean) {
    if (segment_mean.dtype != mt::DType::FP32 ||
        segment_mean.shape.numel() <= 0) {
        return;
    }
    const int64_t n = segment_mean.shape.numel();
    const int64_t d = summary_.shape.d[0];
    const int64_t k = std::min(d, n);
    std::memcpy(summary_.ptr<float>(), segment_mean.ptr<float>(),
                static_cast<size_t>(k) * sizeof(float));
}

const mt::Tensor& PagedPool::summary() const { return summary_; }
int PagedPool::segments() const { return segments_; }
const TierCounters& PagedPool::counters() const {
    return tiers_ ? tiers_->counters() : zero_counters_;
}

void PagedPool::reset_counters() {
    if (tiers_) {
        tiers_->reset_counters();
    }
}

void PagedPool::flush() {
    if (!tiers_) {
        return;
    }
    for (size_t s = 0; s < slots_.size(); ++s) {
        if (slots_[s] < 0) {
            continue;
        }
        mt::Tensor p1, p3, p2;
        row_of(w1_, static_cast<int>(s), w1_.shape.d[1] * w1_.shape.d[2], p1);
        row_of(w3_, static_cast<int>(s), w3_.shape.d[1] * w3_.shape.d[2], p3);
        row_of(w2_, static_cast<int>(s), w2_.shape.d[1] * w2_.shape.d[2], p2);
        tiers_->put(static_cast<int>(uid_[static_cast<size_t>(slots_[s])]), p1, p3,
                    p2, true);
    }
    tiers_->flush();
}

mt::Tensor PagedPool::key_of(const mt::Tensor& w1) const {
    // Dominant input direction of w1 by power iteration (paged.py _key_of).
    const int64_t dff = w1.shape.d[0];
    const int64_t d = w1.shape.d[1];
    mt::Tensor v;
    mt::Shape s;
    s.rank = 1;
    s.d[0] = d;
    v = zeros(s);
    float* vp = v.ptr<float>();
    const float* wp = w1.ptr<float>();
    for (int64_t c = 0; c < d; ++c) {
        double acc = 0.0;
        for (int64_t r = 0; r < dff; ++r) {
            acc += wp[r * d + c];
        }
        vp[c] = static_cast<float>(acc);
    }
    double norm = 0.0;
    for (int64_t c = 0; c < d; ++c) {
        norm += static_cast<double>(vp[c]) * static_cast<double>(vp[c]);
    }
    if (!std::isfinite(norm) || norm < 1e-9) {
        for (int64_t c = 0; c < d; ++c) {
            vp[c] = 1.0f;
        }
        norm = static_cast<double>(d);
    }
    for (int64_t c = 0; c < d; ++c) {
        vp[c] = static_cast<float>(vp[c] / std::sqrt(norm));
    }
    for (int iter = 0; iter < 4; ++iter) {
        std::vector<double> nv(static_cast<size_t>(d), 0.0);
        for (int64_t r = 0; r < dff; ++r) {
            double dot = 0.0;
            for (int64_t c = 0; c < d; ++c) {
                dot += static_cast<double>(wp[r * d + c]) * static_cast<double>(vp[c]);
            }
            for (int64_t c = 0; c < d; ++c) {
                nv[static_cast<size_t>(c)] += dot * static_cast<double>(wp[r * d + c]);
            }
        }
        double nrm = 0.0;
        for (int64_t c = 0; c < d; ++c) {
            nrm += nv[static_cast<size_t>(c)] * nv[static_cast<size_t>(c)];
        }
        if (nrm < 1e-9) {
            break;
        }
        const double scale = 1.0 / std::sqrt(nrm);
        for (int64_t c = 0; c < d; ++c) {
            vp[c] = static_cast<float>(nv[static_cast<size_t>(c)] * scale);
        }
    }
    return v;
}

int PagedPool::build_keys(bool verbose) {
    int done = 0;
    if (!keys_built_ || verbose) {
        mt::Shape s;
        s.rank = 2;
        s.d[0] = n_experts_;
        s.d[1] = d_model_;
        keys_ = zeros(s);
        for (int e = 0; e < n_experts_; ++e) {
            std::vector<mininpz::NpzEntry> entries;
            const std::string f = expert_file_path(e);
            if (!mininpz::read_npz(f, entries)) {
                continue;
            }
            bool got = false;
            for (const auto& en : entries) {
                if (strip_npy(en.name) != "w1") {
                    continue;
                }
                mt::Tensor t = array_to_tensor(en.arr);
                if (t.shape.rank == 2) {
                    mt::Tensor key = key_of(t);
                    std::memcpy(keys_.ptr<float>() + static_cast<size_t>(e) * static_cast<size_t>(d_model_),
                                key.ptr<float>(), static_cast<size_t>(d_model_) * sizeof(float));
                }
                got = true;
            }
            if (got) {
                ++done;
            }
        }
    }
    keys_built_ = true;
    return done;
}

int PagedPool::choose_by_demand(const mt::Tensor& gates_abs,
                                const mt::Tensor& ever_bool,
                                const mt::Tensor& since,
                                const mt::Tensor& last_seen,
                                const std::vector<double>& want,
                                std::vector<int>& out) const {
    out.clear();
    const int k = std::min(resident_, n_experts_);
    if (k <= 0) {
        return 0;
    }
    std::vector<std::pair<double, int>> want_order;
    want_order.reserve(want.size());
    for (size_t e = 0; e < want.size(); ++e) {
        want_order.emplace_back(want[e], static_cast<int>(e));
    }
    std::stable_sort(want_order.begin(), want_order.end(),
                     [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                         if (a.first != b.first) return a.first > b.first;
                         return a.second < b.second;
                     });

    std::vector<int> cur;
    for (int e : slots_) {
        if (e >= 0) {
            cur.push_back(e);
        }
    }
    if (cur.size() < static_cast<size_t>(k)) {  // cold card: fill it in want order
        std::vector<int> filled = cur;
        for (const auto& pr : want_order) {
            if (std::find(filled.begin(), filled.end(), pr.second) == filled.end()) {
                filled.push_back(pr.second);
            }
        }
        if (static_cast<int>(filled.size()) > k) {
            filled.resize(static_cast<size_t>(k));
        }
        out = filled;
        return static_cast<int>(out.size());
    }

    const float* since_p = since.dtype == mt::DType::FP32 ? since.ptr<float>() : nullptr;
    const float* ever_p = ever_bool.dtype == mt::DType::FP32 ? ever_bool.ptr<float>() : nullptr;
    (void)gates_abs;
    (void)last_seen;

    auto weakest_in = [&want](const std::vector<int>& cands) -> int {
        int best = cands[0];
        for (size_t i = 1; i < cands.size(); ++i) {
            int e = cands[i];
            int be = best;
            double we = (e < (int)want.size()) ? want[e] : 0.0;
            double wb = (be < (int)want.size()) ? want[be] : 0.0;
            if (we < wb || (we == wb && e < be)) {
                best = e;
            }
        }
        return best;
    };

    std::vector<int> held = cur;
    std::vector<int> evictable;
    for (int e : cur) {
        const double since_e = (since_p && e < n_experts_) ? since_p[e] : 0.0;
        if (static_cast<double>(segments_) - since_e < static_cast<double>(dwell_)) {
            continue;  // too young to evict
        }
        evictable.push_back(e);
    }

    std::vector<int> plan = cur;
    for (const auto& cand_pr : want_order) {
        const int cand = cand_pr.second;
        if (std::find(held.begin(), held.end(), cand) != held.end()) {
            continue;
        }
        if (evictable.empty()) {
            break;
        }
        const int weakest = weakest_in(evictable);
        const double wweak = (weakest < (int)want.size()) ? want[weakest] : 0.0;
        if (cand_pr.first <= wweak * (1.0 + margin_)) {
            break;
        }
        auto find_it = std::find(plan.begin(), plan.end(), weakest);
        if (find_it != plan.end()) {
            *find_it = cand;
        }
        evictable.erase(std::remove(evictable.begin(), evictable.end(), weakest),
                        evictable.end());
    }

    // terminating cold-start sweep: one never-resident expert, index order.
    std::vector<int> never;
    for (int e = 0; e < n_experts_; ++e) {
        const bool has_ever = ever_p && e < n_experts_ ? ever_p[e] != 0.0f : true;
        if (!has_ever && std::find(plan.begin(), plan.end(), e) == plan.end()) {
            never.push_back(e);
        }
    }
    if (!never.empty() && !evictable.empty()) {
        const int weakest = weakest_in(evictable);
        auto find_it = std::find(plan.begin(), plan.end(), weakest);
        if (find_it != plan.end()) {
            *find_it = never.front();
        }
    }
    if (static_cast<int>(plan.size()) > k) {
        plan.resize(static_cast<size_t>(k));
    }
    out = plan;
    return 0;
}

void PagedPool::set_gate(const mt::Tensor& g) {
    if (g.dtype == mt::DType::FP32 && g.shape.rank == 1 &&
        g.shape.d[0] == n_experts_) {
        gate_ = g;
    }
}

void PagedPool::set_segment_router(const mt::Tensor& w) {
    if (w.dtype == mt::DType::FP32 && w.shape.rank == 2 &&
        w.shape.d[0] >= n_experts_ && w.shape.d[1] == d_model_) {
        segment_router_ = w;
    }
}

mini::JsonValue PagedPool::telemetry() const {
    auto arr_of = [](const mt::Tensor& t, int n,
                     std::map<std::string, mini::JsonValue>& obj, const char* key) {
        std::vector<mini::JsonValue> a;
        const float* p = t.dtype == mt::DType::FP32 ? t.ptr<float>() : nullptr;
        for (int i = 0; i < n; ++i) {
            a.push_back(mini::JsonValue::make_num((double)(p ? p[i] : 0.0f)));
        }
        obj[key] = mini::JsonValue::make_arr(a);
    };
    const int n = n_experts_;
    std::map<std::string, mini::JsonValue> o;
    {
        std::vector<mini::JsonValue> a;
        const float* g = gate_.ptr<float>();
        for (int i = 0; i < n; ++i) {
            double rounded = std::round((double)(double)g[i] * 1e6) / 1e6;
            a.push_back(mini::JsonValue::make_num(rounded));
        }
        o["gate"] = mini::JsonValue::make_arr(a);
        std::vector<mini::JsonValue> gs;
        const float* gs_p = gate_seen_.ptr<float>();
        for (int i = 0; i < n; ++i) {
            double rounded = std::round((double)gs_p[i] * 1e6) / 1e6;
            gs.push_back(mini::JsonValue::make_num(rounded));
        }
        o["gate_seen"] = mini::JsonValue::make_arr(gs);
    }
    arr_of(use_, n, o, "use");
    arr_of(admits_, n, o, "admits");
    arr_of(born_, n, o, "born");
    arr_of(since_, n, o, "since");
    arr_of(last_seen_, n, o, "last_seen");
    {
        std::vector<mini::JsonValue> ev;
        for (int i = 0; i < n; ++i) {
            ev.push_back(mini::JsonValue::make_bool(ever_[static_cast<size_t>(i)] != 0));
        }
        o["ever"] = mini::JsonValue::make_arr(ev);
        std::vector<mini::JsonValue> ids;
        for (int i = 0; i < n; ++i) {
            ids.push_back(mini::JsonValue::make_num(static_cast<double>(uid_[static_cast<size_t>(i)])));
        }
        o["uid"] = mini::JsonValue::make_arr(ids);
    }
    o["next_uid"] = mini::JsonValue::make_num(static_cast<double>(next_uid_));
    o["segments"] = mini::JsonValue::make_num(static_cast<double>(segments_));
    return mini::JsonValue::make_obj(o);
}

void PagedPool::load_telemetry(const mini::JsonValue& t) {
    if (t.t != mini::JsonValue::Type::Obj) {
        return;
    }
    const mini::JsonValue* inner = mini::json_find(t, "telemetry");
    const mini::JsonValue& src = (inner && inner->t == mini::JsonValue::Type::Obj)
                                     ? *inner
                                     : t;
    auto load_arr = [this](const mini::JsonValue& obj, const char* key, float* buf, int n) {
        const mini::JsonValue* v = mini::json_find(obj, key);
        if (!v || v->t != mini::JsonValue::Type::Arr) {
            return;
        }
        const size_t k = std::min<size_t>(v->a.size(), static_cast<size_t>(n));
        for (size_t i = 0; i < k; ++i) {
            double d;
            if (mini::json_try_num(v->a[i], d)) {
                buf[i] = static_cast<float>(d);
            }
        }
    };
    load_arr(src, "use", use_.ptr<float>(), n_experts_);
    load_arr(src, "admits", admits_.ptr<float>(), n_experts_);
    load_arr(src, "born", born_.ptr<float>(), n_experts_);
    load_arr(src, "since", since_.ptr<float>(), n_experts_);
    load_arr(src, "last_seen", last_seen_.ptr<float>(), n_experts_);
    load_arr(src, "gate_seen", gate_seen_.ptr<float>(), n_experts_);

    const mini::JsonValue* ev = mini::json_find(src, "ever");
    if (ev && ev->t == mini::JsonValue::Type::Arr) {
        const size_t k = std::min<size_t>(ev->a.size(), ever_.size());
        for (size_t i = 0; i < k; ++i) {
            ever_[i] = ev->a[i].b ? 1 : 0;
        }
    }
    const mini::JsonValue* uid = mini::json_find(src, "uid");
    if (uid && uid->t == mini::JsonValue::Type::Arr) {
        const size_t k = std::min<size_t>(uid->a.size(), uid_.size());
        for (size_t i = 0; i < k; ++i) {
            double d;
            if (mini::json_try_num(uid->a[i], d)) {
                uid_[i] = static_cast<long long>(d);
            }
        }
    }
    const mini::JsonValue* nu = mini::json_find(src, "next_uid");
    if (nu) {
        double d;
        if (mini::json_try_num(*nu, d)) {
            next_uid_ = static_cast<long long>(d);
        }
    } else {
        next_uid_ = 0;
        for (long long u : uid_) {
            next_uid_ = std::max(next_uid_, u);
        }
        next_uid_ += 1;
    }
    const mini::JsonValue* seg = mini::json_find(src, "segments");
    if (seg) {
        double d;
        if (mini::json_try_num(*seg, d)) {
            segments_ = static_cast<int>(d);
        }
    }
    have_telemetry_ = true;
}

// Tier-2 / store::save_paged wiring.
void PagedPool::lookup_gate(int eid, double& g) const {
    if (gate_.dtype == mt::DType::FP32 && eid >= 0 &&
        eid < static_cast<int>(gate_.shape.d[0])) {
        g = static_cast<double>(gate_.ptr<float>()[static_cast<size_t>(eid)]);
    } else {
        g = 1.0;
    }
}

void PagedPool::emit_manifest_lifecycle(std::map<std::string, mini::JsonValue>& obj) const {
    const int n = n_experts_;
    std::vector<mini::JsonValue> everv, sincev;
    everv.reserve(static_cast<size_t>(n));
    sincev.reserve(static_cast<size_t>(n));
    const float* sp = (since_.dtype == mt::DType::FP32) ? since_.ptr<float>() : nullptr;
    for (int i = 0; i < n; ++i) {
        everv.push_back(mini::JsonValue::make_bool(ever_[static_cast<size_t>(i)] != 0));
        double d = (sp != nullptr) ? static_cast<double>(sp[i]) : 0.0;
        sincev.push_back(mini::JsonValue::make_num(d));
    }
    obj["pool_ever"] = mini::JsonValue::make_arr(everv);
    obj["pool_since"] = mini::JsonValue::make_arr(sincev);
    obj["read_only"] = mini::JsonValue::make_bool(read_only_);
    obj["pool_ram"] = mini::JsonValue::make_num(static_cast<double>(ram_capacity_));
}

// store::save_paged names this telemetry_json (delegates to telemetry()).
mini::JsonValue PagedPool::telemetry_json() const { return telemetry(); }

bool PagedPool::is_moment_key(const std::string& key) {
    const char* p = key.c_str();
    size_t n = key.size();
    if (n >= 2 && (p[n - 2] == '_' && (p[n - 1] == 'm' || p[n - 1] == 'v'))) return true;
    if (n >= 2 && p[n - 2] == '|' && (p[n - 1] == 'm' || p[n - 1] == 'v' || p[n - 1] == 't')) {
        return true;
    }
    return false;
}

bool PagedPool::is_router_key(const std::string& key) {
    if (ends_with(key, "router.weight") || ends_with(key, "depth_emb")) return true;
    return false;
}

// ---- private helpers ------------------------------------------------------

void PagedPool::row_of(const mt::Tensor& t, int slot, int64_t row_numel,
                        mt::Tensor& out) const {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = t.shape.d[1];
    s.d[1] = t.shape.d[2];
    out = zeros(s);
    const float* src = t.ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(row_numel);
    std::memcpy(out.ptr<float>(), src, static_cast<size_t>(row_numel) * sizeof(float));
}

void PagedPool::set_row(mt::Tensor& t, int slot, const mt::Tensor& row) {
    const int64_t row_numel = t.shape.d[1] * t.shape.d[2];
    float* dst = t.ptr<float>() + static_cast<size_t>(slot) * static_cast<size_t>(row_numel);
    std::memcpy(dst, row.ptr<float>(), static_cast<size_t>(row_numel) * sizeof(float));
}

std::string PagedPool::expert_file_path(int uid) const {
    if (experts_dir_set_ && tiers_) {
        return (fs::path(experts_dir_) / expert_file_name(uid)).string();
    }
    return expert_file_name(uid);
}

}  // namespace minagi::paged