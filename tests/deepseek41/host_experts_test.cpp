#include "strata/deepseek41/host_expert_cache.hpp"
#include "strata/deepseek41/expert_file.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
using namespace strata::deepseek41;
namespace fs = std::filesystem;
namespace {
int checks = 0;
void check(bool ok, const char* label) { ++checks; if (!ok) throw std::runtime_error(label); }
void rejects(const std::function<void()>& f, const char* label) {
    bool rejected = false; try { f(); } catch (const std::exception&) { rejected = true; } check(rejected, label);
}
uint64_t budget(const Geometry& g, size_t slots) {
    return size_t(g.layers) * size_t(g.experts) * sizeof(int32_t) + slots * (nvfp4_expert_bytes(g) + 3 * sizeof(int32_t));
}
std::vector<uint8_t> native(const Geometry& g, const fs::path& file) {
    const size_t raw_bytes = size_t(g.hidden) * size_t(g.expert_width) / 32 * 17 * 3;
    std::vector<uint8_t> raw(raw_bytes), result(nvfp4_expert_bytes(g));
    std::ifstream in(file, std::ios::binary); in.exceptions(std::ios::badbit | std::ios::failbit);
    in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
    ExpertView view;
    for (size_t r = 0; r < 3; ++r) { view.parts[r].data = {raw.data() + r * raw_bytes / 3, raw_bytes / 3}; view.parts[r].type = 39; }
    pack_nvfp4_expert(g, view, result); return result;
}
void replay(const Geometry& g, const std::array<fs::path, 3>& files) {
    HostExpertCache cache(g, budget(g, 2)); Mxfp4ExpertFileReader reader(g);
    std::array<std::vector<uint8_t>, 3> expected;
    for (size_t i = 0; i < 3; ++i) expected[i] = native(g, files[i]);
    const std::array<std::pair<int, int>, 3> ids{{{0, 84}, {1, 86}, {0, 327}}};
    std::vector<uint8_t> stage(nvfp4_expert_bytes(g));
    for (int i : {0, 1, 0, 2, 0, 1}) {
        const auto [l, e] = ids[size_t(i)]; const uint8_t* p = cache.find(l, e);
        if (!p) { reader.read(files[size_t(i)].string(), stage); p = cache.insert(l, e, stage); }
        check(std::memcmp(p, expected[size_t(i)].data(), stage.size()) == 0, "cached/direct bytes differ from buffered original adapter");
        check(cache.resident_bytes() <= cache.layout().payload_bytes, "resident payload exceeds fixed capacity");
    }
    const auto cs = cache.stats(); const auto io = reader.stats();
    check(cs.hits == 2 && cs.misses == 4 && cs.evictions == 2, "global LRU cold/warm/eviction sequence");
    check(io.reads == 4 && io.converted == 4 && !io.failures, "cache hits perform no file read or conversion");
    check(io.completed_bytes == 4 * reader.raw_bytes() && io.requested_bytes == 4 * reader.staging_bytes(), "direct IO bytes are separate from aligned request bytes");
    check(cache.layout().allocated_bytes() == budget(g, 2), "payload plus metadata obey exact allowance");
    std::printf("replay hidden=%d hits=%llu misses=%llu evictions=%llu direct_bytes=%llu allocated=%zu staging=%zu\n",
        g.hidden,(unsigned long long)cs.hits,(unsigned long long)cs.misses,(unsigned long long)cs.evictions,
        (unsigned long long)io.completed_bytes,cache.layout().allocated_bytes(),reader.staging_bytes());
}
}
int main(int argc, char** argv) {
    fs::path dir;
    try {
        Geometry g; g.layers = 2; g.hidden = g.expert_width = 64; g.experts = 384; g.top_k = 6;
        const size_t bytes = nvfp4_expert_bytes(g);
        check(host_expert_cache_layout(g, 0).allocated_bytes() == 0, "disabled cache has no heap allocations");
        rejects([&] { host_expert_cache_layout(g, budget(g, 1) - 1); }, "one-byte deficit below a single slot rejected");
        check(host_expert_cache_layout(g, budget(g, 2) - 1).slots == 1, "budget rounded down including metadata");
        check(host_expert_cache_layout(g, std::numeric_limits<uint64_t>::max()).slots == 768, "oversized allowance clamps to unique original pairs");
        auto invalid = g; invalid.layers = 0;
        rejects([&] { HostExpertCache cache(invalid, 0); }, "disabled cache still validates geometry");
        HostExpertCache cache(g, budget(g, 2)); std::vector<uint8_t> a(bytes, 11), b(bytes, 22), c(bytes, 33);
        cache.insert(0, 7, a); cache.insert(1, 7, b);
        check(cache.find(0, 7)[0] == 11 && cache.find(1, 7)[0] == 22, "equal expert IDs on different layers do not alias");
        cache.find(0, 7); cache.insert(0, 8, c);
        check(!cache.find(1, 7) && cache.find(0, 7)[0] == 11, "hit refreshes LRU and eviction removes the old mapping");
        rejects([&] { cache.insert(1, 7, {b.data(), b.size() - 1}); }, "partial insert rejected before eviction");
        check(cache.find(0, 8)[0] == 33 && cache.find(0, 7)[0] == 11, "failed insert preserves cached bytes");
        rejects([&] { cache.find(0, int64_t(1) << 32); }, "wide expert ID rejected without narrowing");
        rejects([&] { cache.find(-1, 7); }, "negative original layer rejected");
        HostExpertCache disabled(g, 0);
        check(!disabled.find(0, 7) && disabled.resident_bytes() == 0, "disabled cache always misses without residency");
        dir = fs::temp_directory_path() / ("strata-host-experts-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir); std::array<fs::path, 3> files;
        const size_t raw_bytes = size_t(g.hidden) * size_t(g.expert_width) / 32 * 17 * 3;
        for (size_t i = 0; i < 3; ++i) {
            files[i] = dir / (std::to_string(i) + ".mxfp4"); std::vector<uint8_t> raw(raw_bytes, uint8_t(0x21 + i));
            for (size_t k = 0; k < raw.size(); k += 17) raw[k] = 127;
            if (!i) raw[0] = 117; // exercises the exact global-scale tail too
            std::ofstream out(files[i], std::ios::binary); out.exceptions(std::ios::badbit | std::ios::failbit);
            out.write(reinterpret_cast<const char*>(raw.data()), std::streamsize(raw.size()));
        }
        replay(g, files);
        Mxfp4ExpertFileReader reader(g); std::vector<uint8_t> result(bytes);
        reader.read(files[0].string(), result); const auto good = result; const auto before = reader.stats();
        rejects([&] { reader.read((dir / "missing").string(), result); }, "missing source is an error");
        fs::resize_file(files[1], raw_bytes - 1);
        rejects([&] { reader.read(files[1].string(), result); }, "truncated source rejected before IO");
        check(result == good && reader.stats().reads == before.reads, "failed source open/size cannot expose stale conversion or publish a read");
        {
            std::fstream bad(files[2], std::ios::binary | std::ios::in | std::ios::out); const char scale = 0; bad.write(&scale, 1);
        }
        rejects([&] { reader.read(files[2].string(), result); }, "inexact MXFP4 scale fails rather than fabricating weights");
        check(reader.stats().converted == 1 && reader.stats().failures == 3, "failed conversion is not counted as a successful expert");
        reader.read(files[0].string(), result);
        check(result == good && reader.stats().converted == 2, "reader can recover using a complete original payload");
        if (argc == 4) {
            g.layers = 40; g.hidden = 5120; g.expert_width = 2304;
            replay(g, {fs::path(argv[1]), fs::path(argv[2]), fs::path(argv[3])});
        } else if (argc != 1) throw std::invalid_argument("Expected no arguments or three original expert files");
        fs::remove_all(dir); std::printf("%d checks, 0 failures\n", checks); return 0;
    } catch (const std::exception& e) {
        if (!dir.empty()) { std::error_code error; fs::remove_all(dir, error); }
        std::fprintf(stderr,"%s\n",e.what()); return 1;
    }
}
