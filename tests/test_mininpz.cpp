#include "mininpz.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static int failures = 0;

#define CHECK(name, cond) do { \
    if (cond) std::cout << "ok " << name << "\n"; \
    else { std::cout << "FAIL " << name << "\n"; ++failures; } \
} while(0)

int main() {
    using namespace mininpz;

    // ---- NPY round-trip float32 (12,) ----
    {
        Shape sh; sh.rank = 1; sh.d[0] = 12;
        std::vector<float> vals(12);
        for (int i = 0; i < 12; ++i) vals[i] = static_cast<float>(i + 1);
        std::string path = "test_npy_f32_12.npy";
        bool w = write_npy(path, DType::F4, sh, vals.data());
        CHECK("npy_write_f32_12", w);
        Array a;
        bool r = read_npy(path, a);
        CHECK("npy_read_f32_12", r);
        CHECK("npy_dtype_f32_12", a.dtype == DType::F4);
        CHECK("npy_shape_f32_12", a.shape == sh);
        CHECK("npy_bytes_f32_12", a.bytes.size() == 12 * 4);
        bool bytes_ok = true;
        for (int i = 0; i < 12; ++i) {
            float v;
            std::memcpy(&v, &a.bytes[i * 4], 4);
            if (std::fabs(v - (i + 1)) > 1e-5f) { bytes_ok = false; break; }
        }
        CHECK("npy_values_f32_12", bytes_ok);
        std::remove(path.c_str());
    }

    // ---- NPY round-trip 2-D (3,4) float32 ----
    {
        Shape sh; sh.rank = 2; sh.d[0] = 3; sh.d[1] = 4;
        std::vector<float> vals(12);
        for (int i = 0; i < 12; ++i) vals[i] = static_cast<float>(i + 1);
        std::string path = "test_npy_2d.npy";
        bool w = write_npy(path, DType::F4, sh, vals.data());
        CHECK("npy_write_2d", w);
        Array a;
        bool r = read_npy(path, a);
        CHECK("npy_read_2d", r);
        CHECK("npy_dtype_2d", a.dtype == DType::F4);
        CHECK("npy_shape_2d", a.shape == sh);
        CHECK("npy_bytes_2d", a.bytes.size() == 12 * 4);
        std::remove(path.c_str());
    }

    // ---- NPY round-trip int16 (5,5) ----
    {
        Shape sh; sh.rank = 2; sh.d[0] = 5; sh.d[1] = 5;
        std::vector<int16_t> vals(25);
        for (int i = 0; i < 25; ++i) vals[i] = static_cast<int16_t>(i);
        std::string path = "test_npy_i16.npy";
        bool w = write_npy(path, DType::I2, sh, vals.data());
        CHECK("npy_write_i16", w);
        Array a;
        bool r = read_npy(path, a);
        CHECK("npy_read_i16", r);
        CHECK("npy_dtype_i16", a.dtype == DType::I2);
        CHECK("npy_shape_i16", a.shape == sh);
        CHECK("npy_bytes_i16", a.bytes.size() == 25 * 2);
        std::remove(path.c_str());
    }

    // ---- NPY golden format ----
    {
        Shape sh; sh.rank = 1; sh.d[0] = 3;
        std::vector<float> vals = {1.0f, 2.0f, 3.0f};
        std::string path = "test_npy_golden.npy";
        bool w = write_npy(path, DType::F4, sh, vals.data());
        CHECK("npy_golden_write", w);
        std::ifstream f(path, std::ios::binary);
        CHECK("npy_golden_open", f.is_open());
        char magic[6];
        f.read(magic, 6);
        bool magic_ok = std::memcmp(magic, "\x93NUMPY", 6) == 0;
        CHECK("npy_golden_magic", magic_ok);
        uint8_t ver[2];
        f.read(reinterpret_cast<char*>(ver), 2);
        CHECK("npy_golden_ver", ver[0] == 1 && ver[1] == 0);
        uint16_t hlen;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        CHECK("npy_golden_hlen_div64", (6 + 2 + 2 + hlen) % 64 == 0);
        std::remove(path.c_str());
    }

    // ---- NPZ round-trip ----
    {
        Shape s1; s1.rank = 2; s1.d[0] = 265; s1.d[1] = 512;
        std::vector<float> v1(265 * 512, 0.5f);
        Shape s2; s2.rank = 2; s2.d[0] = 2048; s2.d[1] = 512;
        std::vector<float> v2(2048 * 512, 0.25f);
        Shape s3; s3.rank = 2; s3.d[0] = 2048; s3.d[1] = 512;
        std::vector<int16_t> v3(2048 * 512, 7);

        std::vector<NpzEntry> entries;
        entries.push_back({"core.tok_emb", {DType::F4, s1, {}}});
        entries.back().arr.bytes.assign(reinterpret_cast<const uint8_t*>(v1.data()),
                                        reinterpret_cast<const uint8_t*>(v1.data()) + v1.size() * 4);
        entries.push_back({"w1", {DType::F4, s2, {}}});
        entries.back().arr.bytes.assign(reinterpret_cast<const uint8_t*>(v2.data()),
                                        reinterpret_cast<const uint8_t*>(v2.data()) + v2.size() * 4);
        entries.push_back({"w1_m", {DType::I2, s3, {}}});
        entries.back().arr.bytes.assign(reinterpret_cast<const uint8_t*>(v3.data()),
                                        reinterpret_cast<const uint8_t*>(v3.data()) + v3.size() * 2);

        std::string path = "test_npz_round.npz";
        bool w = write_npz(path, entries);
        CHECK("npz_write", w);

        std::vector<NpzEntry> read_back;
        bool r = read_npz(path, read_back);
        CHECK("npz_read", r);
        CHECK("npz_count", read_back.size() == 3);
        CHECK("npz_name0", read_back[0].name == "core.tok_emb");
        CHECK("npz_name1", read_back[1].name == "w1");
        CHECK("npz_name2", read_back[2].name == "w1_m");
        CHECK("npz_dtype0", read_back[0].arr.dtype == DType::F4);
        CHECK("npz_dtype1", read_back[1].arr.dtype == DType::F4);
        CHECK("npz_dtype2", read_back[2].arr.dtype == DType::I2);
        CHECK("npz_shape0", read_back[0].arr.shape == s1);
        CHECK("npz_shape1", read_back[1].arr.shape == s2);
        CHECK("npz_shape2", read_back[2].arr.shape == s3);
        CHECK("npz_bytes0", read_back[0].arr.bytes == entries[0].arr.bytes);
        CHECK("npz_bytes1", read_back[1].arr.bytes == entries[1].arr.bytes);
        CHECK("npz_bytes2", read_back[2].arr.bytes == entries[2].arr.bytes);
        std::remove(path.c_str());
    }

    // ---- JSON parse ----
    {
        std::string txt = R"({"expert":3,"name":"e00003.npz","gate":0.42,"ok":true,"opt":null,"rows":[1,2,3]})";
        mini::JsonValue v;
        bool p = mini::json_parse(txt, v);
        CHECK("json_parse", p);
        CHECK("json_obj", v.t == mini::JsonValue::Type::Obj);
        const mini::JsonValue* expert = mini::json_find(v, "expert");
        CHECK("json_expert_exists", expert != nullptr);
        double ev = 0;
        CHECK("json_expert_val", expert && mini::json_try_num(*expert, ev) && std::fabs(ev - 3.0) < 1e-9);
        const mini::JsonValue* name = mini::json_find(v, "name");
        std::string ns;
        CHECK("json_name_val", name && mini::json_try_str(*name, ns) && ns == "e00003.npz");
        const mini::JsonValue* gate = mini::json_find(v, "gate");
        double gv = 0;
        CHECK("json_gate_val", gate && mini::json_try_num(*gate, gv) && std::fabs(gv - 0.42) < 1e-9);
        const mini::JsonValue* ok = mini::json_find(v, "ok");
        CHECK("json_ok_val", ok && ok->t == mini::JsonValue::Type::Bool && ok->b == true);
        const mini::JsonValue* opt = mini::json_find(v, "opt");
        CHECK("json_opt_val", opt && opt->t == mini::JsonValue::Type::Null);
        const mini::JsonValue* rows = mini::json_find(v, "rows");
        CHECK("json_rows_val", rows && rows->t == mini::JsonValue::Type::Arr && rows->a.size() == 3);
        CHECK("json_rows_0", rows && rows->a.size() > 0 && std::fabs(rows->a[0].n - 1.0) < 1e-9);
        CHECK("json_rows_1", rows && rows->a.size() > 1 && std::fabs(rows->a[1].n - 2.0) < 1e-9);
        CHECK("json_rows_2", rows && rows->a.size() > 2 && std::fabs(rows->a[2].n - 3.0) < 1e-9);

        // round-trip dump + reparse
        std::string dumped = mini::json_dump(v, 0);
        mini::JsonValue v2;
        bool p2 = mini::json_parse(dumped, v2);
        CHECK("json_roundtrip_parse", p2);
        const mini::JsonValue* e2 = mini::json_find(v2, "expert");
        double ev2 = 0;
        CHECK("json_roundtrip_expert", e2 && mini::json_try_num(*e2, ev2) && std::fabs(ev2 - 3.0) < 1e-9);

        // \uXXXX escape
        std::string esc = R"("\u00e9")";
        mini::JsonValue ev3;
        bool p3 = mini::json_parse(esc, ev3);
        CHECK("json_escape_parse", p3);
        std::string es;
        CHECK("json_escape_val", ev3.t == mini::JsonValue::Type::Str && mini::json_try_str(ev3, es) && es == "\xc3\xa9");
    }

    // ---- Atomicity ----
    {
        Shape sh; sh.rank = 1; sh.d[0] = 4;
        std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<NpzEntry> entries;
        entries.push_back({"a", {DType::F4, sh, {}}});
        entries.back().arr.bytes.assign(reinterpret_cast<const uint8_t*>(vals.data()),
                                        reinterpret_cast<const uint8_t*>(vals.data()) + 16);
        std::string path = "test_npz_atomic.npz";
        bool w = write_npz(path, entries);
        CHECK("npz_atomic_write", w);
        std::ifstream f(path);
        CHECK("npz_atomic_exists", f.good());
        std::ifstream tmp(path + ".tmp.npz");
        CHECK("npz_atomic_no_tmp", !tmp.good());
        std::remove(path.c_str());
    }

    if (failures == 0) {
        std::cout << "ALL_TESTS_PASSED\n";
        return 0;
    } else {
        std::cout << "SOME_TESTS_FAILED\n";
        return 1;
    }
}
