#include "tokenizer.hpp"
#include <cstring>
namespace minagi {
int ByteTokenizer::id_of_special(std::string_view s) {
  for (size_t i = 0; i < kSpecials.size(); ++i) {
    size_t l = std::strlen(kSpecials[i]);
    if (s.size() == l && std::memcmp(s.data(), kSpecials[i], l) == 0)
      return static_cast<int>(256 + i);
  }
  return -1;
}
std::string_view ByteTokenizer::special_of_id(int id) {
  if (id < 256 || id > 264) return {};
  return kSpecials[static_cast<size_t>(id - 256)];
}
static bool match_special(std::string_view utf8, size_t pos, const char* sp, size_t sp_len) {
  return utf8.size() - pos >= sp_len && std::memcmp(utf8.data() + pos, sp, sp_len) == 0;
}
std::vector<int32_t> ByteTokenizer::encode(std::string_view utf8) const {
  std::vector<int32_t> ids;
  size_t pos = 0;
  while (pos < utf8.size()) {
    bool matched = false;
    for (size_t i = 0; i < kSpecials.size(); ++i) {
      size_t l = std::strlen(kSpecials[i]);
      if (match_special(utf8, pos, kSpecials[i], l)) {
        ids.push_back(static_cast<int32_t>(256 + i));
        pos += l;
        matched = true;
        break;
      }
    }
    if (!matched) {
      ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(utf8[pos])));
      ++pos;
    }
  }
  return ids;
}
namespace {
bool is_cont(unsigned char c) { return (c & 0xC0) == 0x80; }
std::string utf8_decode_replace(std::string_view bytes) {
  std::string out;
  size_t i = 0;
  unsigned char repl[3] = {0xEF, 0xBF, 0xBD};
  auto bad = [&](size_t& j) {
    out.append(reinterpret_cast<char*>(repl), 3);
    ++j;
  };
  while (i < bytes.size()) {
    unsigned char b = static_cast<unsigned char>(bytes[i]);
    if (b < 0x80) {
      out.push_back(static_cast<char>(b));
      ++i;
    } else if (b >= 0xC2 && b <= 0xDF) {
      if (i + 1 < bytes.size() && is_cont(static_cast<unsigned char>(bytes[i + 1]))) {
        out.push_back(static_cast<char>(b));
        out.push_back(bytes[i + 1]);
        i += 2;
      } else {
        bad(i);
      }
    } else if (b >= 0xE0 && b <= 0xEF) {
      if (i + 2 < bytes.size() && is_cont(static_cast<unsigned char>(bytes[i + 1])) &&
          is_cont(static_cast<unsigned char>(bytes[i + 2]))) {
        unsigned char b2 = static_cast<unsigned char>(bytes[i + 1]);
        if ((b == 0xE0 && b2 < 0xA0) || (b == 0xED && b2 > 0x9F)) {
          bad(i);
        } else {
          out.push_back(static_cast<char>(b));
          out.push_back(bytes[i + 1]);
          out.push_back(bytes[i + 2]);
          i += 3;
        }
      } else {
        bad(i);
      }
    } else if (b >= 0xF0 && b <= 0xF4) {
      if (i + 3 < bytes.size() && is_cont(static_cast<unsigned char>(bytes[i + 1])) &&
          is_cont(static_cast<unsigned char>(bytes[i + 2])) &&
          is_cont(static_cast<unsigned char>(bytes[i + 3]))) {
        unsigned char b2 = static_cast<unsigned char>(bytes[i + 1]);
        if ((b == 0xF0 && b2 < 0x90) || (b == 0xF4 && b2 > 0x8F)) {
          bad(i);
        } else {
          out.push_back(static_cast<char>(b));
          out.push_back(bytes[i + 1]);
          out.push_back(bytes[i + 2]);
          out.push_back(bytes[i + 3]);
          i += 4;
        }
      } else {
        bad(i);
      }
    } else {
      bad(i);
    }
  }
  return out;
}
}  // namespace
std::string ByteTokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string out;
  std::string buf;
  for (int32_t id : ids) {
    if (id >= 256 && id <= 264) {
      if (!buf.empty()) {
        out += utf8_decode_replace(buf);
        buf.clear();
      }
      out += special_of_id(id);
    } else if (id >= 0 && id < 256) {
      buf.push_back(static_cast<char>(static_cast<unsigned char>(id)));
    }
  }
  if (!buf.empty()) out += utf8_decode_replace(buf);
  return out;
}
int ByteTokenizer::token_to_id(std::string_view token) const {
  int sp = id_of_special(token);
  if (sp != -1) return sp;
  if (token.size() == 1) {
    return static_cast<int>(static_cast<unsigned char>(token[0]));
  }
  return -1;
}
}  // namespace minagi