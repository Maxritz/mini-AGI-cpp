#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
namespace minagi {
struct ByteTokenizer {
  static constexpr size_t kSize = 256 + 9;
  static constexpr std::array<const char*, 9> kSpecials = {
      "\x3cthink\x3e", "\x3c\x2fthink\x3e", "<user>", "</user>",
      "<bot>", "</bot>", "<g>", "</g>", "<|endoftext|>"};
  static int id_of_special(std::string_view s);
  static std::string_view special_of_id(int id);
  std::vector<int32_t> encode(std::string_view utf8) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  size_t size() const { return kSize; }
  int token_to_id(std::string_view token) const;
};
}  // namespace minagi