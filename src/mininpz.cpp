#include "mininpz.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace mininpz {

// ---- Shape ----
size_t Shape::numel() const {
    size_t n = 1;
    for (size_t i = 0; i < rank; ++i) n *= d[i];
    return n;
}

bool Shape::operator==(const Shape& o) const {
    if (rank != o.rank) return false;
    for (size_t i = 0; i < rank; ++i)
        if (d[i] != o.d[i]) return false;
    return true;
}

size_t Shape::offset(const std::array<size_t, 4>& idx) const {
    size_t off = 0;
    size_t stride = 1;
    for (size_t i = rank; i-- > 0;) {
        off += idx[i] * stride;
        stride *= d[i];
    }
    return off;
}

// ---- DType helpers ----
static const char* dtype_descr(DType dt) {
    switch (dt) {
        case DType::U8: return "|u1";
        case DType::I2: return "<i2";
        case DType::I4: return "<i4";
        case DType::I8: return "<i8";
        case DType::F4: return "<f4";
        case DType::F8: return "<f8";
    }
    return "|u1";
}

static bool descr_to_dtype(const std::string& d, DType& out) {
    if (d == "|u1") { out = DType::U8; return true; }
    if (d == "<i2") { out = DType::I2; return true; }
    if (d == "<i4") { out = DType::I4; return true; }
    if (d == "<i8") { out = DType::I8; return true; }
    if (d == "<f4") { out = DType::F4; return true; }
    if (d == "<f8") { out = DType::F8; return true; }
    return false;
}

static size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::U8: return 1;
        case DType::I2: return 2;
        case DType::I4: return 4;
        case DType::I8: return 8;
        case DType::F4: return 4;
        case DType::F8: return 8;
    }
    return 1;
}

// ---- NPY read ----
static bool parse_header_dict(const std::string& h, DType& dt, Shape& sh) {
    // Minimal parser for: {'descr': '<f4', 'fortran_order': False, 'shape': (12,), }
    size_t i = 0;
    auto skip_ws = [&]() {
        while (i < h.size() && (h[i] == ' ' || h[i] == '\t' || h[i] == '\n' || h[i] == '\r')) ++i;
    };
    auto expect = [&](char c) {
        skip_ws();
        if (i >= h.size() || h[i] != c) return false;
        ++i;
        return true;
    };
    auto parse_str = [&]() -> std::string {
        skip_ws();
        if (i >= h.size() || h[i] != '\'') return "";
        ++i;
        std::string r;
        while (i < h.size() && h[i] != '\'') {
            if (h[i] == '\\' && i + 1 < h.size()) { r += h[i + 1]; i += 2; }
            else { r += h[i++]; }
        }
        if (i < h.size()) ++i;
        return r;
    };
    auto parse_bool = [&]() -> bool {
        skip_ws();
        if (h.compare(i, 4, "True") == 0) { i += 4; return true; }
        if (h.compare(i, 5, "False") == 0) { i += 5; return false; }
        return false;
    };
    auto parse_tuple = [&]() -> std::vector<size_t> {
        std::vector<size_t> r;
        if (!expect('(')) return r;
        skip_ws();
        while (i < h.size() && h[i] != ')') {
            std::string num;
            while (i < h.size() && h[i] >= '0' && h[i] <= '9') num += h[i++];
            if (!num.empty()) r.push_back(std::stoull(num));
            skip_ws();
            if (i < h.size() && h[i] == ',') ++i;
            skip_ws();
        }
        expect(')');
        return r;
    };

    if (!expect('{')) return false;
    skip_ws();
    while (i < h.size() && h[i] != '}') {
        std::string key = parse_str();
        if (!expect(':')) return false;
        skip_ws();
        if (key == "descr") {
            std::string v = parse_str();
            if (!descr_to_dtype(v, dt)) return false;
        } else if (key == "fortran_order") {
            parse_bool();
        } else if (key == "shape") {
            auto dims = parse_tuple();
            sh.rank = dims.size();
            for (size_t k = 0; k < dims.size() && k < 4; ++k) sh.d[k] = dims[k];
        } else {
            // skip value
            if (h[i] == '\'') parse_str();
            else if (h[i] == '(') parse_tuple();
            else {
                while (i < h.size() && h[i] != ',' && h[i] != '}') ++i;
            }
        }
        skip_ws();
        if (i < h.size() && h[i] == ',') ++i;
        skip_ws();
    }
    expect('}');
    return true;
}

bool read_npy(const std::string& path, Array& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6];
    if (!f.read(magic, 6) || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t major, minor;
    if (!f.read(reinterpret_cast<char*>(&major), 1)) return false;
    if (!f.read(reinterpret_cast<char*>(&minor), 1)) return false;
    if (major != 1 && major != 2) return false;
    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t hl;
        if (!f.read(reinterpret_cast<char*>(&hl), 2)) return false;
        hlen = hl;
    } else {
        if (!f.read(reinterpret_cast<char*>(&hlen), 4)) return false;
    }
    std::string header(hlen, '\0');
    if (!f.read(&header[0], hlen)) return false;
    // trim trailing spaces and newline
    while (!header.empty() && (header.back() == ' ' || header.back() == '\n' || header.back() == '\r'))
        header.pop_back();
    DType dt;
    Shape sh;
    if (!parse_header_dict(header, dt, sh)) return false;
    size_t sz = sh.numel() * dtype_size(dt);
    out.dtype = dt;
    out.shape = sh;
    out.bytes.resize(sz);
    if (sz > 0) {
        if (!f.read(reinterpret_cast<char*>(out.bytes.data()), sz)) return false;
    }
    return true;
}

// ---- NPY write ----
static std::string build_header(DType dt, const Shape& sh) {
    std::ostringstream os;
    os << "{'descr': '" << dtype_descr(dt) << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < sh.rank; ++i) {
        if (i) os << ", ";
        os << sh.d[i];
    }
    if (sh.rank == 1) os << ",";
    os << "), }";
    std::string dict = os.str();
    // pad so the full npy header block (magic 6 + ver 2 + len 2 + header) is
    // divisible by 64 and the header ends with a newline
    size_t prefix = 6 + 2 + 2;
    size_t total = prefix + dict.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    dict += std::string(pad, ' ') + "\n";
    return dict;
}

bool write_npy(const std::string& path, DType dt, const Shape& sh, const void* data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("\x93NUMPY", 6);
    uint8_t ver[2] = {1, 0};
    f.write(reinterpret_cast<const char*>(ver), 2);
    std::string header = build_header(dt, sh);
    uint16_t hlen = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(header.data(), header.size());
    size_t sz = sh.numel() * dtype_size(dt);
    if (sz > 0 && data) f.write(reinterpret_cast<const char*>(data), sz);
    return f.good();
}

// ---- ZIP helpers ----
static uint32_t crc32_table[256];
static bool crc32_init = false;

static void init_crc32() {
    if (crc32_init) return;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc32_table[i] = c;
    }
    crc32_init = true;
}

static uint32_t crc32_update(uint32_t c, const uint8_t* data, size_t len) {
    init_crc32();
    for (size_t i = 0; i < len; ++i)
        c = crc32_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c;
}

static void write_u16(std::ostream& os, uint16_t v) {
    uint8_t b[2] = {static_cast<uint8_t>(v & 0xFF), static_cast<uint8_t>((v >> 8) & 0xFF)};
    os.write(reinterpret_cast<const char*>(b), 2);
}

static void write_u32(std::ostream& os, uint32_t v) {
    uint8_t b[4] = {
        static_cast<uint8_t>(v & 0xFF),
        static_cast<uint8_t>((v >> 8) & 0xFF),
        static_cast<uint8_t>((v >> 16) & 0xFF),
        static_cast<uint8_t>((v >> 24) & 0xFF)
    };
    os.write(reinterpret_cast<const char*>(b), 4);
}

static void write_u64(std::ostream& os, uint64_t v) {
    write_u32(os, static_cast<uint32_t>(v & 0xFFFFFFFFu));
    write_u32(os, static_cast<uint32_t>(v >> 32));
}

static uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

static uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint64_t read_u64(const uint8_t* p) {
    uint64_t lo = read_u32(p);
    uint64_t hi = read_u32(p + 4);
    return lo | (hi << 32);
}

// Parse a ZIP64 extended-info extra field (tag 0x0001). Fields (in order)
// are present only for values that were set to the sentinel: for local
// headers: uncompressed size, compressed size. For central directory:
// uncompressed size, compressed size, local header offset, disk number.
// out order: [0]=uncomp, [1]=comp, [2]=local_off. Returns number of fields read.
static int parse_zip64_extra(const uint8_t* extra, size_t elen,
                             uint64_t out[3]) {
    out[0] = out[1] = out[2] = 0;
    size_t i = 0;
    int count = 0;
    while (i + 4 <= elen) {
        uint16_t tag = read_u16(extra + i);
        uint16_t sz = read_u16(extra + i + 2);
        i += 4;
        if (i + sz > elen) return count;
        if (tag == 0x0001) {
            size_t j = i;
            // up to 4 x 8-byte fields: uncomp, comp, local offset, disk
            for (int f = 0; f < 3 && j + 8 <= i + sz; ++f) {
                out[f] = read_u64(extra + j);
                j += 8;
            }
            return 3;
        }
        i += sz;
    }
    return count;
}

// ---- NPZ write ----
// Always emitted in ZIP64 form (unconditional): 0xFFFFFFFF sentinels plus a
// 0x0001 extra field holding the 8-byte values. Python/numpy read this fine,
// and it keeps the writer independent of whether cumulative offsets cross the
// 4 GiB boundary (which the >4 GiB model stores do).
bool write_npz(const std::string& path, const std::vector<NpzEntry>& entries) {
    std::string tmp = path + ".tmp.npz";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) return false;

        struct EntryInfo {
            std::string name;
            uint64_t local_off;
            uint64_t comp_size;
            uint64_t uncomp_size;
            uint32_t crc;
        };
        std::vector<EntryInfo> infos;

        for (const auto& e : entries) {
            // prefix: magic(6) + ver(2) + hlen(2) + header
            std::string prefix;
            prefix.reserve(10 + 128);
            prefix.append("\x93NUMPY", 6);
            prefix.push_back(1);  // major
            prefix.push_back(0);  // minor
            std::string header = build_header(e.arr.dtype, e.arr.shape);
            uint16_t hlen = static_cast<uint16_t>(header.size());
            prefix.push_back(static_cast<char>(hlen & 0xFF));
            prefix.push_back(static_cast<char>(hlen >> 8));
            prefix += header;
            const size_t data_len = e.arr.shape.numel() * dtype_size(e.arr.dtype);
            const uint64_t zsize = static_cast<uint64_t>(prefix.size()) + data_len;
            uint32_t crc = crc32_update(0xFFFFFFFFu,
                                        reinterpret_cast<const uint8_t*>(prefix.data()),
                                        prefix.size());
            if (data_len > 0)
                crc = crc32_update(crc, e.arr.bytes.data(), data_len);
            crc ^= 0xFFFFFFFFu;
            const uint64_t local_off = static_cast<uint64_t>(f.tellp());

            // local file header (zip64)
            write_u32(f, 0x04034b50);
            write_u16(f, 45);  // version needed
            write_u16(f, 0);   // flags
            write_u16(f, 0);   // method (stored)
            write_u16(f, 0);   // mod time
            write_u16(f, 0);   // mod date
            write_u32(f, crc);
            write_u32(f, 0xFFFFFFFFu); // comp size (zip64)
            write_u32(f, 0xFFFFFFFFu); // uncomp size (zip64)
            write_u16(f, static_cast<uint16_t>(e.name.size()));
            write_u16(f, 20);  // extra len (0x0001, 16 bytes payload)
            f.write(e.name.data(), e.name.size());
            uint8_t extra[20];
            extra[0] = 0x01; extra[1] = 0x00;  // tag 0x0001
            extra[2] = 0x10; extra[3] = 0x00;  // size 16
            uint8_t* p = extra + 4;
            for (int k = 0; k < 8; ++k) *p++ = static_cast<uint8_t>((zsize >> (8 * k)) & 0xFF);
            for (int k = 0; k < 8; ++k) *p++ = static_cast<uint8_t>((zsize >> (8 * k)) & 0xFF);
            f.write(reinterpret_cast<const char*>(extra), 20);
            f.write(prefix.data(), prefix.size());
            if (data_len > 0)
                f.write(reinterpret_cast<const char*>(e.arr.bytes.data()), data_len);

            infos.push_back({e.name, local_off, zsize, zsize, crc});
        }

        const uint64_t cd_off = static_cast<uint64_t>(f.tellp());
        for (const auto& info : infos) {
            write_u32(f, 0x02014b50);
            write_u16(f, 45);  // version made by
            write_u16(f, 45);  // version needed
            write_u16(f, 0);   // flags
            write_u16(f, 0);   // method
            write_u16(f, 0);   // mod time
            write_u16(f, 0);   // mod date
            write_u32(f, info.crc);
            write_u32(f, 0xFFFFFFFFu); // comp size (zip64)
            write_u32(f, 0xFFFFFFFFu); // uncomp size (zip64)
            write_u16(f, static_cast<uint16_t>(info.name.size()));
            write_u16(f, 28);  // extra len (0x0001, 24 bytes payload)
            write_u16(f, 0);   // comment
            write_u16(f, 0);   // disk number start
            write_u16(f, 0);   // internal attrs
            write_u32(f, 0);   // external attrs
            write_u32(f, 0xFFFFFFFFu); // local header offset (zip64)
            f.write(info.name.data(), info.name.size());
            uint8_t extra[28];
            extra[0] = 0x01; extra[1] = 0x00;  // tag 0x0001
            extra[2] = 0x18; extra[3] = 0x00;  // size 24
            uint8_t* p = extra + 4;
            auto put64 = [&](uint64_t v) {
                for (int k = 0; k < 8; ++k) *p++ = static_cast<uint8_t>((v >> (8 * k)) & 0xFF);
            };
            put64(info.uncomp_size);
            put64(info.comp_size);
            put64(info.local_off);
            f.write(reinterpret_cast<const char*>(extra), 28);
        }
        const uint64_t cd_size = static_cast<uint64_t>(f.tellp()) - cd_off;

        // ZIP64 end of central directory record
        const uint64_t z64_off = cd_off + cd_size;
        write_u32(f, 0x06064b50);
        write_u64(f, 44);  // size of remaining record (44 bytes)
        write_u16(f, 45);  // version made by
        write_u16(f, 45);  // version needed
        write_u32(f, 0);   // this disk
        write_u32(f, 0);   // cd start disk
        write_u64(f, static_cast<uint64_t>(infos.size())); // entries this disk
        write_u64(f, static_cast<uint64_t>(infos.size())); // entries total
        write_u64(f, cd_size);
        write_u64(f, cd_off);

        // ZIP64 locator
        write_u32(f, 0x07064b50);
        write_u32(f, 0);       // disk with zip64 eocd
        write_u64(f, z64_off); // offset of zip64 eocd
        write_u32(f, 1);       // total disks

        // standard EOCD (closing sentinels; used only as the search anchor)
        write_u32(f, 0x06054b50);
        write_u16(f, 0);
        write_u16(f, 0);
        write_u16(f, 0xFFFF);
        write_u16(f, 0xFFFF);
        write_u32(f, 0xFFFFFFFFu);
        write_u32(f, 0xFFFFFFFFu);
        write_u16(f, 0);
    }

    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) return false;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

// ---- NPZ read ----
// Streaming, seek-based reader. Handles both the legacy 32-bit layout (old
// writer / small archives) and ZIP64 (offsets and sizes > 4 GiB). Only stored
// (uncompressed) entries are supported, which is what every writer emits.
bool read_npz(const std::string& path, std::vector<NpzEntry>& out) {
    out.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const int64_t file_size = f.tellg();
    if (file_size < 22) return false;

    // locate the standard EOCD by scanning its 4-byte signature at the end
    const int64_t max_scan = std::min<int64_t>(file_size, 65557 + 22);
    const int64_t rd_start = file_size - max_scan;
    f.seekg(static_cast<std::streamoff>(rd_start));
    std::vector<uint8_t> tail(static_cast<size_t>(max_scan));
    if (!f.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(max_scan)))
        return false;
    int64_t eocd = -1;
    for (int64_t i = max_scan - 22; i >= 0; --i) {
        if (tail[static_cast<size_t>(i)] == 0x50 &&
            tail[static_cast<size_t>(i + 1)] == 0x4b &&
            tail[static_cast<size_t>(i + 2)] == 0x05 &&
            tail[static_cast<size_t>(i + 3)] == 0x06) {
            eocd = rd_start + i;
            break;
        }
    }
    if (eocd < 0) return false;
    const size_t eoff = static_cast<size_t>(eocd - rd_start);

    uint64_t num_entries = read_u16(&tail[eoff + 10]);
    const uint32_t cd_size32 = read_u32(&tail[eoff + 12]);
    const uint32_t cd_off32 = read_u32(&tail[eoff + 16]);
    uint64_t cd_off = cd_off32;

    if (num_entries == 0xFFFF || cd_size32 == 0xFFFFFFFFu || cd_off32 == 0xFFFFFFFFu) {
        // ZIP64: locator sits 20 bytes before the standard EOCD.
        const int64_t zloc = eocd - 20;
        if (zloc < 0) return false;
        f.seekg(static_cast<std::streamoff>(zloc));
        uint8_t lb[20];
        if (!f.read(reinterpret_cast<char*>(lb), 20)) return false;
        if (!(lb[0] == 0x50 && lb[1] == 0x4b && lb[2] == 0x06 && lb[3] == 0x07)) return false;
        const uint64_t z64_off = read_u64(lb + 8);
        if (z64_off + 56 > static_cast<uint64_t>(file_size)) return false;
        f.seekg(static_cast<std::streamoff>(z64_off));
        uint8_t z[56];
        if (!f.read(reinterpret_cast<char*>(z), 56)) return false;
        if (!(z[0] == 0x50 && z[1] == 0x4b && z[2] == 0x06 && z[3] == 0x06)) return false;
        num_entries = read_u64(z + 32);
        cd_off = read_u64(z + 48);
    }

    uint64_t cd_cursor = cd_off;
    for (uint64_t i = 0; i < num_entries; ++i) {
        f.seekg(static_cast<std::streamoff>(cd_cursor));
        uint8_t c[46];
        if (!f.read(reinterpret_cast<char*>(c), 46)) return false;
        if (!(c[0] == 0x50 && c[1] == 0x4b && c[2] == 0x01 && c[3] == 0x02)) return false;
        const uint16_t name_len = read_u16(c + 28);
        const uint16_t extra_len = read_u16(c + 30);
        const uint16_t comment_len = read_u16(c + 32);
        cd_cursor += 46 + name_len + extra_len + comment_len;
        const uint32_t c_comp = read_u32(c + 20);
        const uint32_t c_uncomp = read_u32(c + 24);
        const uint32_t lo32 = read_u32(c + 42);

        std::string name(static_cast<size_t>(name_len), '\0');
        if (name_len > 0 && !f.read(&name[0], name_len)) return false;
        std::vector<uint8_t> extra(static_cast<size_t>(extra_len));
        if (extra_len > 0 && !f.read(reinterpret_cast<char*>(extra.data()), extra_len))
            return false;
        if (comment_len > 0) f.seekg(static_cast<std::streamoff>(comment_len), std::ios::cur);

        uint64_t local_off = lo32;
        if (c_uncomp == 0xFFFFFFFFu || c_comp == 0xFFFFFFFFu || lo32 == 0xFFFFFFFFu) {
            uint64_t z64[3] = {0, 0, 0};
            parse_zip64_extra(extra.data(), extra.size(), z64);
            int fld = 0;
            if (c_uncomp == 0xFFFFFFFFu) ++fld;  // uncompressed size field
            if (c_comp == 0xFFFFFFFFu) ++fld;    // compressed size field
            if (lo32 == 0xFFFFFFFFu) local_off = z64[fld++];
        }

        f.seekg(static_cast<std::streamoff>(local_off));
        uint8_t lh[30];
        if (!f.read(reinterpret_cast<char*>(lh), 30)) return false;
        if (!(lh[0] == 0x50 && lh[1] == 0x4b && lh[2] == 0x03 && lh[3] == 0x04)) return false;
        const uint16_t l_method = read_u16(lh + 8);
        const uint32_t l_comp = read_u32(lh + 18);
        const uint32_t l_uncomp = read_u32(lh + 22);
        const uint16_t l_name = read_u16(lh + 26);
        const uint16_t l_extra = read_u16(lh + 28);
        uint64_t lcomp = l_comp;
        if (l_comp == 0xFFFFFFFFu || l_uncomp == 0xFFFFFFFFu) {
            if (l_extra > 0) {
                f.seekg(static_cast<std::streamoff>(l_name), std::ios::cur);
                std::vector<uint8_t> lex(static_cast<size_t>(l_extra));
                if (!f.read(reinterpret_cast<char*>(lex.data()), l_extra)) return false;
                uint64_t z64[3] = {0, 0, 0};
                parse_zip64_extra(lex.data(), lex.size(), z64);
                int fld = 0;
                if (l_uncomp == 0xFFFFFFFFu) ++fld;  // uncompressed size field
                if (l_comp == 0xFFFFFFFFu) lcomp = z64[fld++];
            }
        }
        const uint64_t data_off = local_off + 30 + l_name + l_extra;
        if (l_method != 0) return false;  // only stored
        if (data_off + lcomp > static_cast<uint64_t>(file_size)) return false;

        f.seekg(static_cast<std::streamoff>(data_off));
        uint8_t nm[10];
        if (!f.read(reinterpret_cast<char*>(nm), 10)) return false;
        if (std::memcmp(nm, "\x93NUMPY", 6) != 0) return false;
        const uint8_t major = nm[6];
        if (major != 1 && major != 2) return false;
        uint32_t hlen = 0;
        if (major == 1) {
            hlen = read_u16(nm + 8);
        } else {
            hlen = read_u32(nm + 6);
        }
        std::string header(hlen, '\0');
        if (hlen > 0 && !f.read(&header[0], hlen)) return false;
        while (!header.empty() && (header.back() == ' ' || header.back() == '\n' || header.back() == '\r'))
            header.pop_back();
        DType dt;
        Shape sh;
        if (!parse_header_dict(header, dt, sh)) return false;
        const size_t sz = sh.numel() * dtype_size(dt);

        Array arr;
        arr.dtype = dt;
        arr.shape = sh;
        arr.bytes.resize(sz);
        if (sz > 0 &&
            !f.read(reinterpret_cast<char*>(arr.bytes.data()), static_cast<std::streamsize>(sz)))
            return false;

        out.push_back({name, arr});
    }
    return true;
}

} // namespace mininpz