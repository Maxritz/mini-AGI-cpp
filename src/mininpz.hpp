#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace mininpz {

enum class DType : uint8_t { U8, I2, I4, I8, F4, F8 };

struct Shape {
    size_t rank = 0;
    std::array<size_t, 4> d{};
    size_t numel() const;
    bool operator==(const Shape& o) const;
    size_t offset(const std::array<size_t, 4>& idx) const;
};

struct Array {
    DType dtype;
    Shape shape;
    std::vector<uint8_t> bytes;
};

struct NpzEntry {
    std::string name;
    Array arr;
};

bool read_npy(const std::string& path, Array& out);
bool write_npy(const std::string& path, DType dt, const Shape& sh, const void* data);
bool write_npz(const std::string& path, const std::vector<NpzEntry>& entries);
bool read_npz(const std::string& path, std::vector<NpzEntry>& out);

} // namespace mininpz

namespace mini {

struct JsonValue;
using JsonObj = std::map<std::string, JsonValue>;
using JsonArr = std::vector<JsonValue>;

struct JsonValue {
    enum class Type { Null, Bool, Num, Str, Arr, Obj } t = Type::Null;
    bool b = false;
    double n = 0;
    std::string s;
    JsonArr a;
    JsonObj o;

    static JsonValue null();
    static JsonValue make_bool(bool v);
    static JsonValue make_num(double v);
    static JsonValue make_str(std::string v);
    static JsonValue make_arr(JsonArr v);
    static JsonValue make_obj(JsonObj v);
};

bool json_parse(const std::string& text, JsonValue& out);
std::string json_dump(const JsonValue& v, int indent_every_level_exists_or_compact = 0);
const JsonValue* json_find(const JsonValue& obj, const char* key);
bool json_try_num(const JsonValue& v, double& out);
bool json_try_str(const JsonValue& v, std::string& out);

} // namespace mini
