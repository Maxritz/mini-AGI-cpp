#include "store.hpp"
#include "mininpz.hpp"
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
namespace fs = std::filesystem;
int main() {
    fs::path wdir = "tests/golden_wavef/weights";
    std::cout << "exists=" << fs::exists(wdir) << "\n";
    std::cout << "core=" << fs::exists(wdir/"core.npz") << " routers=" << fs::exists(wdir/"routers.npz") << "\n";
    std::cout << "experts_dir=" << fs::exists(wdir/"experts") << "\n";
    int idx=0;
    for (auto& p : fs::directory_iterator(wdir/"experts")) { std::cout << "exp " << idx << ": " << p.path().filename() << " size=" << fs::file_size(p) << "\n"; ++idx; }
    ::store::Manifest man;
    std::map<std::string, mininpz::Array> state;
    bool ok = ::store::load(wdir.string(), man, state);
    std::cout << "load=" << ok << " n_experts=" << man.n_experts << " state_size=" << state.size() << "\n";
    if (!ok) return 1;
    // try reading an expert npz directly
    std::vector<mininpz::NpzEntry> ents;
    bool r = mininpz::read_npz((wdir/"experts"/"e00000.npz").string(), ents);
    std::cout << "read_npz e00000=" << r << " nentries=" << ents.size() << "\n";
    for (auto& e : ents) std::cout << "  entry: " << e.name << " dtype=" << static_cast<int>(e.arr.dtype) << " rank=" << e.arr.shape.rank << "\n";
    return 0;
}
