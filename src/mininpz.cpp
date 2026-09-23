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

static uint32_t crc32(const uint8_t* data, size_t len) {
    init_crc32();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        c = crc32_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
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
bool write_npz(const std::string& path, const std::vector<NpzEntry>& entries) {
    std::string tmp = path + ".tmp.npz";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return false;

    struct EntryInfo {
        std::string name;
        uint32_t local_off;
        uint32_t comp_size;
        uint32_t uncomp_size;
        uint32_t crc;
    };
    std::vector<EntryInfo> infos;

    for (const auto& e : entries) {
        // serialize array to npy bytes
        std::ostringstream npy_os;
        npy_os.write("\x93NUMPY", 6);
        uint8_t ver[2] = {1, 0};
        npy_os.write(reinterpret_cast<const char*>(ver), 2);
        std::string header = build_header(e.arr.dtype, e.arr.shape);
        uint16_t hlen = static_cast<uint16_t>(header.size());
        npy_os.write(reinterpret_cast<const char*>(&hlen), 2);
        npy_os.write(header.data(), header.size());
        size_t sz = e.arr.shape.numel() * dtype_size(e.arr.dtype);
        if (sz > 0) npy_os.write(reinterpret_cast<const char*>(e.arr.bytes.data()), sz);
        std::string npy_data = npy_os.str();

        uint32_t crc = crc32(reinterpret_cast<const uint8_t*>(npy_data.data()), npy_data.size());
        uint32_t local_off = static_cast<uint32_t>(f.tellp());

        // local file header
        write_u32(f, 0x04034b50);
        write_u16(f, 20); // version
        write_u16(f, 0);  // flags
        write_u16(f, 0);  // method (stored)
        write_u16(f, 0);  // mod time
        write_u16(f, 0);  // mod date
        write_u32(f, crc);
        write_u32(f, static_cast<uint32_t>(npy_data.size())); // comp size
        write_u32(f, static_cast<uint32_t>(npy_data.size())); // uncomp size
        write_u16(f, static_cast<uint16_t>(e.name.size()));
        write_u16(f, 0); // extra
        f.write(e.name.data(), e.name.size());
        f.write(npy_data.data(), npy_data.size());

        infos.push_back({e.name, local_off, static_cast<uint32_t>(npy_data.size()),
                         static_cast<uint32_t>(npy_data.size()), crc});
    }

    uint32_t cd_off = static_cast<uint32_t>(f.tellp());
    for (const auto& info : infos) {
        write_u32(f, 0x02014b50);
        write_u16(f, 20); // version made by
        write_u16(f, 20); // version needed
        write_u16(f, 0);  // flags
        write_u16(f, 0);  // method
        write_u16(f, 0);  // mod time
        write_u16(f, 0);  // mod date
        write_u32(f, info.crc);
        write_u32(f, info.comp_size);
        write_u32(f, info.uncomp_size);
        write_u16(f, static_cast<uint16_t>(info.name.size()));
        write_u16(f, 0); // extra
        write_u16(f, 0); // comment
        write_u16(f, 0); // disk number start
        write_u16(f, 0); // internal attrs
        write_u32(f, 0); // external attrs
        write_u32(f, info.local_off);
        f.write(info.name.data(), info.name.size());
    }
    uint32_t cd_size = static_cast<uint32_t>(f.tellp()) - cd_off;

    // end of central directory
    write_u32(f, 0x06054b50);
    write_u16(f, 0); // disk number
    write_u16(f, 0); // disk with cd
    write_u16(f, static_cast<uint16_t>(infos.size()));
    write_u16(f, static_cast<uint16_t>(infos.size()));
    write_u32(f, cd_size);
    write_u32(f, cd_off);
    write_u16(f, 0); // comment length

    f.close();

    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) return false;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

// ---- NPZ read ----
bool read_npz(const std::string& path, std::vector<NpzEntry>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> buf(std::istreambuf_iterator<char>(f), {});
    if (buf.size() < 22) return false;

    // find end of central directory
    size_t eocd = 0;
    if (buf.size() >= 22) {
        for (size_t i = buf.size() - 21U; i-- > 0;) {
            if (buf[i] == 0x50 && buf[i + 1] == 0x4b && buf[i + 2] == 0x05 && buf[i + 3] == 0x06) {
                eocd = i;
                break;
            }
        }
    }
    if (eocd == 0 && !(buf[0] == 0x50 && buf[1] == 0x4b && buf[2] == 0x05 && buf[3] == 0x06)) return false;

    uint16_t num_entries = read_u16(&buf[eocd + 10]);
    uint32_t cd_size = read_u32(&buf[eocd + 12]);
    uint32_t cd_off = read_u32(&buf[eocd + 16]);

    // ZIP64 end of central directory fallback
    if (num_entries == 0xFFFF || cd_size == 0xFFFFFFFFu || cd_off == 0xFFFFFFFFu) {
        if (eocd < 20) return false;
        size_t zloc = eocd - 20;
        if (buf[zloc] == 0x50 && buf[zloc + 1] == 0x4b && buf[zloc + 2] == 0x06 && buf[zloc + 3] == 0x07) {
            uint64_t z64_off = read_u64(&buf[zloc + 8]);
            if (z64_off + 56 > buf.size()) return false;
            const uint8_t* z = &buf[static_cast<size_t>(z64_off)];
            if (!(z[0] == 0x50 && z[1] == 0x4b && z[2] == 0x06 && z[3] == 0x06)) return false;
            uint64_t ents = read_u64(z + 32);
            uint64_t cdsz = read_u64(z + 40);
            uint64_t cdo = read_u64(z + 48);
            if (ents > 0xFFFF) return false;
            num_entries = static_cast<uint16_t>(ents);
            cd_size = static_cast<uint32_t>(cdsz);
            cd_off = static_cast<uint32_t>(cdo);
        } else {
            return false;
        }
    }

    size_t pos = cd_off;
    for (uint16_t i = 0; i < num_entries; ++i) {
        if (pos + 46 > buf.size()) return false;
        if (buf[pos] != 0x50 || buf[pos + 1] != 0x4b || buf[pos + 2] != 0x01 || buf[pos + 3] != 0x02)
            return false;
        uint16_t name_len = read_u16(&buf[pos + 28]);
        uint16_t extra_len = read_u16(&buf[pos + 30]);
        uint16_t comment_len = read_u16(&buf[pos + 32]);
        uint32_t c_comp_size = read_u32(&buf[pos + 20]);
        uint32_t c_uncomp_size = read_u32(&buf[pos + 24]);
        uint32_t local_off = read_u32(&buf[pos + 42]);
        if (name_len + comment_len > 0 && pos + 46 + name_len + extra_len + comment_len > buf.size())
            return false;
        std::string name(reinterpret_cast<const char*>(&buf[pos + 46]), name_len);
        if (c_comp_size == 0xFFFFFFFFu || c_uncomp_size == 0xFFFFFFFFu || local_off == 0xFFFFFFFFu) {
            if (pos + 46 + name_len + extra_len > buf.size()) return false;
            uint64_t z64[3];
            parse_zip64_extra(&buf[pos + 46 + name_len], extra_len, z64);
            uint32_t u = 0, c = 0, lo = 0;
            // fields present in order: uncomp, comp, local offset
            for (int idx = 0; idx < 3; ++idx) {
                if (idx == 0) { if (c_uncomp_size == 0xFFFFFFFFu) u = static_cast<uint32_t>(z64[0]); }
                if (idx == 1) { if (c_comp_size == 0xFFFFFFFFu) c = static_cast<uint32_t>(z64[1]); }
                if (idx == 2) { if (local_off == 0xFFFFFFFFu) lo = static_cast<uint32_t>(z64[2]); }
            }
            if (u != 0) c_uncomp_size = u;
            if (c != 0) c_comp_size = c;
            if (lo != 0) local_off = lo;
        }
        pos += 46 + name_len + extra_len + comment_len;

        if (local_off + 30 > buf.size()) return false;
        if (buf[local_off] != 0x50 || buf[local_off + 1] != 0x4b || buf[local_off + 2] != 0x03 || buf[local_off + 3] != 0x04)
            return false;
        uint16_t l_method = read_u16(&buf[local_off + 8]);
        uint32_t l_comp_size = read_u32(&buf[local_off + 18]);
        uint32_t l_uncomp_size = read_u32(&buf[local_off + 22]);
        uint16_t l_name_len = read_u16(&buf[local_off + 26]);
        uint16_t l_extra_len = read_u16(&buf[local_off + 28]);
        if (l_comp_size == 0xFFFFFFFFu || l_uncomp_size == 0xFFFFFFFFu) {
            if (local_off + 30 + l_name_len + l_extra_len > buf.size()) return false;
            uint64_t z64[3];
            parse_zip64_extra(&buf[local_off + 30 + l_name_len], l_extra_len, z64);
            if (l_uncomp_size == 0xFFFFFFFFu) l_uncomp_size = static_cast<uint32_t>(z64[0]);
            if (l_comp_size == 0xFFFFFFFFu) l_comp_size = static_cast<uint32_t>(z64[1]);
        }
        size_t data_off = local_off + 30 + l_name_len + l_extra_len;

        if (l_method != 0) return false; // only stored
        if (data_off + l_comp_size > buf.size()) return false;

        Array arr;
        std::vector<uint8_t> npy_bytes(buf.begin() + data_off, buf.begin() + data_off + l_comp_size);
        // parse npy from memory
        if (npy_bytes.size() < 10) return false;
        if (std::memcmp(npy_bytes.data(), "\x93NUMPY", 6) != 0) return false;
        uint8_t major = npy_bytes[6];
        if (major != 1 && major != 2) return false;
        size_t hoff = 8;
        uint32_t hlen = 0;
        if (major == 1) {
            hlen = read_u16(&npy_bytes[hoff]);
            hoff += 2;
        } else {
            hlen = read_u32(&npy_bytes[hoff]);
            hoff += 4;
        }
        if (hoff + hlen > npy_bytes.size()) return false;
        std::string header(reinterpret_cast<const char*>(&npy_bytes[hoff]), hlen);
        while (!header.empty() && (header.back() == ' ' || header.back() == '\n' || header.back() == '\r'))
            header.pop_back();
        DType dt;
        Shape sh;
        if (!parse_header_dict(header, dt, sh)) return false;
        arr.dtype = dt;
        arr.shape = sh;
        size_t sz = sh.numel() * dtype_size(dt);
        if (hoff + hlen + sz > npy_bytes.size()) return false;
        arr.bytes.assign(npy_bytes.begin() + hoff + hlen, npy_bytes.begin() + hoff + hlen + sz);

        out.push_back({name, arr});
    }
    return true;
}

} // namespace mininpz
