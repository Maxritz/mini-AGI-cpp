#include "mininpz.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace mini {

JsonValue JsonValue::null() { return JsonValue{}; }
JsonValue JsonValue::make_bool(bool v) {
  JsonValue j;
  j.t = Type::Bool;
  j.b = v;
  return j;
}
JsonValue JsonValue::make_num(double v) {
  JsonValue j;
  j.t = Type::Num;
  j.n = v;
  return j;
}
JsonValue JsonValue::make_str(std::string v) {
  JsonValue j;
  j.t = Type::Str;
  j.s = std::move(v);
  return j;
}
JsonValue JsonValue::make_arr(JsonArr v) {
  JsonValue j;
  j.t = Type::Arr;
  j.a = std::move(v);
  return j;
}
JsonValue JsonValue::make_obj(JsonObj v) {
  JsonValue j;
  j.t = Type::Obj;
  j.o = std::move(v);
  return j;
}

namespace {

struct Parser {
  const std::string& s;
  size_t i = 0;

  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  bool peek(char c) {
    ws();
    return i < s.size() && s[i] == c;
  }
  bool parse_value(JsonValue& v) {
    ws();
    if (i >= s.size()) return false;
    switch (s[i]) {
      case '{':
        return parse_obj(v);
      case '[':
        return parse_arr(v);
      case '"':
        return parse_str(v);
      case 't':
        return lit("true") && (v = JsonValue::make_bool(true), true);
      case 'f':
        return lit("false") && (v = JsonValue::make_bool(false), true);
      case 'n':
        return lit("null") && (v = JsonValue::null(), true);
      default:
        return parse_num(v);
    }
  }
  bool lit(const char* w) {
    size_t l = std::strlen(w);
    if (i + l > s.size()) return false;
    if (std::strncmp(s.c_str() + i, w, l) != 0) return false;
    i += l;
    return true;
  }
  bool parse_obj(JsonValue& v) {
    if (s[i++] != '{') return false;
    JsonObj o;
    ws();
    if (i < s.size() && s[i] == '}') {
      ++i;
      v = JsonValue::make_obj(std::move(o));
      return true;
    }
    for (;;) {
      JsonValue key;
      ws();
      if (!parse_str(key)) return false;
      ws();
      if (i >= s.size() || s[i] != ':') return false;
      ++i;
      JsonValue val;
      if (!parse_value(val)) return false;
      o[std::move(key.s)] = std::move(val);
      ws();
      if (i >= s.size()) return false;
      if (s[i] == ',') {
        ++i;
        continue;
      }
      if (s[i] == '}') {
        ++i;
        break;
      }
      return false;
    }
    v = JsonValue::make_obj(std::move(o));
    return true;
  }
  bool parse_arr(JsonValue& v) {
    if (s[i++] != '[') return false;
    JsonArr a;
    ws();
    if (i < s.size() && s[i] == ']') {
      ++i;
      v = JsonValue::make_arr(std::move(a));
      return true;
    }
    for (;;) {
      JsonValue val;
      if (!parse_value(val)) return false;
      a.push_back(std::move(val));
      ws();
      if (i >= s.size()) return false;
      if (s[i] == ',') {
        ++i;
        continue;
      }
      if (s[i] == ']') {
        ++i;
        break;
      }
      return false;
    }
    v = JsonValue::make_arr(std::move(a));
    return true;
  }
  bool parse_str(JsonValue& v) {
    if (s[i++] != '"') return false;
    std::string out;
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"') {
        v = JsonValue::make_str(std::move(out));
        return true;
      }
      if (c == '\\') {
        if (i >= s.size()) return false;
        char e = s[i++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            if (i + 4 > s.size()) return false;
            unsigned cp = 0;
            for (int k = 0; k < 4; ++k) {
              char h = s[i++];
              cp <<= 4;
              if (h >= '0' && h <= '9') cp |= (h - '0');
              else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
              else return false;
            }
            // encode as UTF-8; surrogate pairs pass through as two 3-byte encodings
            char seq[4];
            int n = 0;
            if (cp < 0x80) {
              seq[n++] = static_cast<char>(cp);
            } else if (cp < 0x800) {
              seq[n++] = static_cast<char>(0xC0 | (cp >> 6));
              seq[n++] = static_cast<char>(0x80 | (cp & 0x3F));
            } else {
              seq[n++] = static_cast<char>(0xE0 | (cp >> 12));
              seq[n++] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
              seq[n++] = static_cast<char>(0x80 | (cp & 0x3F));
            }
            out.append(seq, n);
            break;
          }
          default: return false;
        }
      } else {
        out.push_back(c);
      }
    }
    return false;
  }
  bool parse_num(JsonValue& v) {
    size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.'
                            || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
      ++i;
    }
    if (i == start) return false;
    errno = 0;
    char* end = nullptr;
    double d = std::strtod(s.c_str() + start, &end);
    if (end != s.c_str() + i) return false;
    v = JsonValue::make_num(d);
    return true;
  }
};

struct Dumper {
  std::string out;
  const std::string& str(const JsonValue& v) {
    out.push_back('"');
    for (char c : v.s) {
      switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
          if (static_cast<unsigned char>(c) < 0x20) {
            char b[7];
            std::snprintf(b, sizeof(b), "\\u%04x", c & 0xFF);
            out += b;
          } else {
            out.push_back(c);
          }
      }
    }
    out.push_back('"');
    return out;
  }
  void go(const JsonValue& v, int depth, int indent) {
    switch (v.t) {
      case JsonValue::Type::Null: out += "null"; break;
      case JsonValue::Type::Bool: out += v.b ? "true" : "false"; break;
      case JsonValue::Type::Num: {
        char b[64];
        if (std::floor(v.n) == v.n && std::fabs(v.n) < 1e15) {
          std::snprintf(b, sizeof(b), "%.0f", v.n);
        } else {
          std::snprintf(b, sizeof(b), "%.17g", v.n);
        }
        out += b;
        break;
      }
      case JsonValue::Type::Str: str(v); break;
      case JsonValue::Type::Arr: {
        out.push_back('[');
        bool first = true;
        for (const auto& e : v.a) {
          if (!first) out.push_back(',');
          first = false;
          if (indent > 0) { out.push_back('\n'); out.append(static_cast<size_t>(indent * (depth + 1)), ' '); }
          go(e, depth + 1, indent);
        }
        if (indent > 0 && !v.a.empty()) { out.push_back('\n'); out.append(static_cast<size_t>(indent * depth), ' '); }
        out.push_back(']');
        break;
      }
      case JsonValue::Type::Obj: {
        out.push_back('{');
        bool first = true;
        for (const auto& kv : v.o) {
          if (!first) out.push_back(',');
          first = false;
          if (indent > 0) { out.push_back('\n'); out.append(static_cast<size_t>(indent * (depth + 1)), ' '); }
          out.push_back('"');
          out += kv.first;
          out.push_back('"');
          out.push_back(':');
          if (indent > 0) out.push_back(' ');
          go(kv.second, depth + 1, indent);
        }
        if (indent > 0 && !v.o.empty()) { out.push_back('\n'); out.append(static_cast<size_t>(indent * depth), ' '); }
        out.push_back('}');
        break;
      }
    }
  }
};

}  // namespace

bool json_parse(const std::string& text, JsonValue& out) {
  Parser p{text};
  if (!p.parse_value(out)) return false;
  p.ws();
  return p.i >= text.size();
}

std::string json_dump(const JsonValue& v, int indent) {
  Dumper d;
  d.go(v, 0, indent);
  return std::move(d.out);
}

const JsonValue* json_find(const JsonValue& obj, const char* key) {
  if (obj.t != JsonValue::Type::Obj) return nullptr;
  auto it = obj.o.find(key);
  return it == obj.o.end() ? nullptr : &it->second;
}

bool json_try_num(const JsonValue& v, double& out) {
  if (v.t != JsonValue::Type::Num) return false;
  out = v.n;
  return true;
}

bool json_try_str(const JsonValue& v, std::string& out) {
  if (v.t != JsonValue::Type::Str) return false;
  out = v.s;
  return true;
}

}  // namespace mini