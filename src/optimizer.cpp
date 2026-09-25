// src/optimizer.cpp
// Implementation of AdamW optimizer for mini-AGI training.

#include "optimizer.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace minagi {

AdamW::AdamW(float lr, float weight_decay, float beta1, float beta2,
             float eps, float clip, int64_t step_count)
    : lr_(lr), weight_decay_(weight_decay), beta1_(beta1), beta2_(beta2),
      eps_(eps), clip_(clip), step_count_(step_count), t_(0) {}

void AdamW::step(const std::vector<std::pair<std::string, mt::Tensor*>>& params_grads)
{
    // Build the paired view for the internal step
    std::vector<std::pair<std::string, std::pair<mt::Tensor*, mt::Tensor*>>> paired;
    for (const auto& pg : params_grads) {
        // We assume grad is stored alongside param; for this simplified interface,
        // we treat pg.second as the param and look up grad from moments_ as zero init.
        // In practice, callers should use the explicit paired version.
        paired.emplace_back(pg.first, std::make_pair(pg.second, nullptr));
    }
    step(paired);
}

void AdamW::step(const std::vector<std::pair<std::string, std::pair<mt::Tensor*, mt::Tensor*>>>& params_grads)
{
    ++t_;
    step_count_ = t_;

    // Phase 1: Global gradient clipping
    if (clip_ > 0.0f) {
        // Compute global L2 norm across all gradients
        double global_norm_sq = 0.0;
        for (const auto& kv : params_grads) {
            mt::Tensor* grad = kv.second.second;
            if (!grad || grad->shape.rank == 0) continue;
            for (int64_t i = 0; i < grad->shape.numel(); ++i) {
                float val = grad->atf(i);
                global_norm_sq += static_cast<double>(val) * static_cast<double>(val);
            }
        }
        float global_norm = static_cast<float>(std::sqrt(global_norm_sq));
        if (global_norm > clip_) {
            float scale = clip_ / global_norm;
            for (auto& kv : params_grads) {
                mt::Tensor* grad = kv.second.second;
                if (!grad || grad->shape.rank == 0) continue;
                for (int64_t i = 0; i < grad->shape.numel(); ++i) {
                    float val = grad->atf(i);
                    grad->set_flat(i, static_cast<double>(val) * scale);
                }
            }
        }
    }

    // Phase 2: Update each parameter
    for (const auto& kv : params_grads) {
        const std::string& key = kv.first;
        mt::Tensor* param = kv.second.first;
        mt::Tensor* grad = kv.second.second;
        if (!param || !grad || param->shape.rank == 0) continue;
        update_param(*param, *grad, key);
    }
}

void AdamW::update_param(mt::Tensor& param, mt::Tensor& grad,
                         const std::string& key)
{
    // Ensure moments are initialized
    auto it = moments_.find(key);
    if (it == moments_.end()) {
        MomentState state;
        state.m = mt::make_zeros(param.shape, mt::DType::FP32);
        state.v = mt::make_zeros(param.shape, mt::DType::FP32);
        it = moments_.try_emplace(key, std::move(state)).first;
    }

    MomentState& ms = it->second;
    const float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    const float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));
    const int64_t numel = param.shape.numel();
    float* m_data = ms.m.ptr<float>();
    float* v_data = ms.v.ptr<float>();

    for (int64_t i = 0; i < numel; ++i) {
        float g = grad.atf(i);
        float p = param.atf(i);

        // Update moments: m = beta1 * m + (1 - beta1) * g
        float m_new = beta1_ * m_data[i] + (1.0f - beta1_) * g;
        // v = beta2 * v + (1 - beta2) * g^2
        float v_new = beta2_ * v_data[i] + (1.0f - beta2_) * g * g;

        m_data[i] = m_new;
        v_data[i] = v_new;

        // Bias-corrected estimates
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;

        // AdamW update (decoupled weight decay)
        float denom = std::sqrt(v_hat) + eps_;
        float update = m_hat / denom;

        // Decoupled weight decay
        if (weight_decay_ != 0.0f) {
            p = p - lr_ * weight_decay_ * p;
        }

        param.set_flat(i, static_cast<double>(p - lr_ * update));
    }
}

void AdamW::clip_gradients(const std::vector<std::pair<std::string, mt::Tensor*>>& params_grads)
{
    clip_ = 1.0f;
    step(params_grads);
}

void AdamW::clip_gradients(const std::vector<std::pair<std::string, std::pair<mt::Tensor*, mt::Tensor*>>>& params_grads)
{
    clip_ = 1.0f;
    step(params_grads);
}

mt::Tensor* AdamW::get_moment_m(const std::string& key)
{
    auto it = moments_.find(key);
    if (it == moments_.end()) return nullptr;
    return &it->second.m;
}

mt::Tensor* AdamW::get_moment_v(const std::string& key)
{
    auto it = moments_.find(key);
    if (it == moments_.end()) return nullptr;
    return &it->second.v;
}

std::vector<std::pair<std::string, mt::Tensor>> AdamW::state_dict() const
{
    std::vector<std::pair<std::string, mt::Tensor>> out;
    // Always include the step count
    // step count stored as a scalar tensor
    {
        mt::Shape s;
        s.rank = 0;
        mt::Tensor step_tensor = mt::make(s, mt::DType::FP32, static_cast<float>(t_));
        out.emplace_back("step", step_tensor);
    }

    for (const auto& kv : moments_) {
        const std::string& key = kv.first;
        const MomentState& ms = kv.second;
        out.emplace_back(key + "_m", ms.m);
        out.emplace_back(key + "_v", ms.v);
    }
    return out;
}

int AdamW::load_state_dict(const std::vector<std::pair<std::string, mt::Tensor>>& sd)
{
    int count = 0;
    for (const auto& kv : sd) {
        const std::string& key = kv.first;
        // Check if this is a moment key: ends with _m or _v, or |m, |v, |t
        if (!minagi::paged::PagedPool::is_moment_key(key)) {
            continue;
        }

        // Strip the suffix to get the parameter name
        std::string param_key;
        if (key.size() >= 2 && key[key.size() - 2] == '_') {
            char suffix = key[key.size() - 1];
            param_key = key.substr(0, key.size() - 2);
            if (suffix == 'm') {
                auto& ms = moments_[param_key];
                if (ms.m.shape.rank == 0) ms.m = kv.second;
                ++count;
            } else if (suffix == 'v') {
                auto& ms = moments_[param_key];
                if (ms.v.shape.rank == 0) ms.v = kv.second;
                ++count;
            }
        } else if (key.size() >= 2 && key[key.size() - 2] == '|') {
            char suffix = key[key.size() - 1];
            param_key = key.substr(0, key.size() - 2);
            if (suffix == 'm') {
                auto& ms = moments_[param_key];
                if (ms.m.shape.rank == 0) ms.m = kv.second;
                ++count;
            } else if (suffix == 'v') {
                auto& ms = moments_[param_key];
                if (ms.v.shape.rank == 0) ms.v = kv.second;
                ++count;
            } else if (suffix == 't') {
                // Step count
                if (kv.second.shape.rank == 0) {
                    t_ = static_cast<int64_t>(kv.second.atf(0));
                    step_count_ = t_;
                }
                ++count;
            }
        }
    }
    return count;
}

} // namespace minagi
