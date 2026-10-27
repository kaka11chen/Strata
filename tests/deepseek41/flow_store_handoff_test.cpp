// Exercise the real patched native ExpertStore, without a model or GPU.
// Build: g++ -std=c++17 -O1 -g -pthread -fsanitize=address,undefined
//   -I<step7q-upstream>/src/dsv41 tests/deepseek41/flow_store_handoff_test.cpp -Wl,--wrap=pread -o <test>
#include "ds4_store.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

using strata::ds4::ExpertStore;
using R = ExpertStore::Retain;
namespace os = strata::ds4::os;

// Pause a real native pread in this Linux test only, to exercise admission while a fill is still in flight.
std::mutex read_mu;
std::condition_variable read_cv;
int gated_fd = -1;
bool read_entered = false, allow_read = false;
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* dst, size_t bytes, off_t off) {
    {
        std::unique_lock<std::mutex> lk(read_mu);
        if (fd == gated_fd) {
            read_entered = true;
            read_cv.notify_all();
            if (!read_cv.wait_for(lk, std::chrono::seconds(5), [] { return allow_read; })) std::abort();
        }
    }
    return __real_pread(fd, dst, bytes, off);
}

struct Fixture {
    ExpertStore::LayerGeo geo;
    std::vector<uint8_t> data[3];
    const uint8_t* parts[3];
    size_t sizes[3] = {137, 4103, 8211};  // unequal quantized payloads and unaligned file offsets
    Fixture() {
        geo.n_exp = 128;
        uint64_t end = 0;
        for (int r = 0; r < 3; ++r) {
            geo.file[r] = 0;
            geo.off[r] = 13 + r * 19001;
            geo.stride[r] = 300001;
            geo.bytes[r] = sizes[r];
            geo.tr[r] = end;
            end += sizes[r] + 8192;
            data[r].resize(sizes[r]);
            for (size_t i = 0; i < sizes[r]; ++i) data[r][i] = (uint8_t) (i * 17 + r * 59);
            parts[r] = data[r].data();
        }
        geo.blob = end;
    }
    size_t slot_bytes() const { return (geo.blob + 2 * ExpertStore::kSlack + 4095) & ~size_t{4095}; }
    ExpertStore::Cfg cfg(size_t slots, uint64_t& available) const {
        ExpertStore::Cfg c;
        c.threads = 1;
        c.budget_bytes = slots * slot_bytes();
        c.floor_bytes = 1 << 20;
        c.ram_avail = [&available] { return available; };
        return c;
    }
    void equal(const ExpertStore& store, int slot) const {
        assert(slot >= 0);
        for (int r = 0; r < 3; ++r) assert(std::memcmp(store.mat(slot, r), parts[r], sizes[r]) == 0);
    }
};

int main() {
    Fixture f;
    uint64_t available = 1ull << 30;
    const std::vector<os::fd_t> missing{os::kBadFd};
    {
        ExpertStore store;
        auto c = f.cfg(2, available);
        assert(store.init(c, {f.geo}, missing, missing));
        assert(store.retain_free(-1, 0, f.parts, f.sizes) == R::invalid);
        assert(store.retain_free(1, 0, f.parts, f.sizes) == R::invalid);
        assert(store.retain_free(0, -1, f.parts, f.sizes) == R::invalid);
        assert(store.retain_free(0, f.geo.n_exp, f.parts, f.sizes) == R::invalid);
        assert(store.retain_free(0, 0, nullptr, f.sizes) == R::invalid);
        assert(store.retain_free(0, 0, f.parts, nullptr) == R::invalid);
        size_t bad_sizes[3] = {f.sizes[0], f.sizes[1] - 1, f.sizes[2]};
        assert(store.retain_free(0, 0, f.parts, bad_sizes) == R::invalid);
        const uint8_t* bad_parts[3] = {f.parts[0], nullptr, f.parts[2]};
        assert(store.retain_free(0, 0, bad_parts, f.sizes) == R::invalid);
        assert(store.find(0, 0) == -1);
        assert(store.retain_free(0, 0, f.parts, f.sizes) == R::stored);
        const int first = store.find(0, 0);
        f.equal(store, first);
        const uint8_t* borrowed = store.blob(first);
        std::vector<uint8_t> owned(borrowed, borrowed + f.geo.blob);
        // Staging is reusable as soon as retain_free returns; the cache owns its copy.
        for (auto& v : f.data) for (uint8_t& x : v) x ^= 0xff;
        assert(std::memcmp(borrowed, owned.data(), owned.size()) == 0);
        assert(store.retain_free(0, 0, f.parts, f.sizes) == R::present);
        assert(std::memcmp(borrowed, owned.data(), owned.size()) == 0);
        const int e = 0;
        std::vector<char> ok;
        assert(store.ensure(0, &e, 1, ok) && ok[0]);
        assert(store.stats().hits == 1 && store.stats().bytes_read == 0 && store.stats().fails == 0);
        assert(store.retain_free(0, 1, f.parts, f.sizes) == R::stored);
        f.equal(store, store.find(0, 1));
        assert(store.retain_free(0, 2, f.parts, f.sizes) == R::no_room);
        store.release(0, &e, 1, ok);
        // Even an unpinned blob borrowed by the flow is not evicted to make room.
        assert(store.retain_free(0, 2, f.parts, f.sizes) == R::no_room);
        assert(store.find(0, 0) == first && store.blob(first) == borrowed);
        assert(std::memcmp(borrowed, owned.data(), owned.size()) == 0);
        assert(store.stats().evictions == 0 && store.stats().bytes_read == 0);
        assert(store.slots_allocated() == 2 && store.slots_allocated() * store.slot_bytes() <= c.budget_bytes);
    }
    std::puts("PASS exact spans, invalid inputs, independent ownership, duplicate, zero-read ensure, pinned/borrowed preservation, full budget");
    {
        ExpertStore store;
        auto c = f.cfg(65, available);
        assert(store.init(c, {f.geo}, missing, missing));
        assert(store.slots_allocated() == 64);
        available = c.floor_bytes + f.slot_bytes() - 1;
        assert(store.retain_free(0, 0, f.parts, f.sizes) == R::no_room);
        assert(store.find(0, 0) == -1);
        available = 1ull << 30;
        for (int e = 0; e < 64; ++e) assert(store.retain_free(0, e, f.parts, f.sizes) == R::stored);
        available = c.floor_bytes;
        assert(store.retain_free(0, 64, f.parts, f.sizes) == R::no_room);
        assert(store.slots_allocated() == 64);
        available = 1ull << 30;
        assert(store.retain_free(0, 64, f.parts, f.sizes) == R::stored);
        assert(store.slots_allocated() == 65);
        assert(store.retain_free(0, 65, f.parts, f.sizes) == R::no_room);
        for (int e = 0; e < 65; ++e) f.equal(store, store.find(0, e));
        assert(store.stats().bytes_read == 0);
    }
    std::puts("PASS floor on free pages and growth, final partial chunk, exact capacity");
    {
        ExpertStore store;
        auto c = f.cfg(128, available);
        int mapped = 0;
        c.map_chunk = [&mapped](void* p, size_t) { ++mapped; return p; };  // no CUDA call in this test
        c.reserve = 0;
        // Only the initial chunk fits the floor during init/pregrow.
        available = c.floor_bytes + 64 * f.slot_bytes();
        c.ram_avail = [&available] { const auto n = available; available = 0; return n; };
        assert(store.init(c, {f.geo}, missing, missing));
        assert(store.slots_allocated() == 64 && mapped == 1);
        for (int e = 0; e < 64; ++e) {
            available = 1ull << 30;
            assert(store.retain_free(0, e, f.parts, f.sizes) == R::stored);
        }
        std::vector<ExpertStore::Vis> visible;
        store.sync(visible, [](int, int) { return true; });
        assert(visible.size() == 64 && store.visible_count() == 64);
        available = 1ull << 30;
        assert(store.retain_free(0, 64, f.parts, f.sizes) == R::no_room);
        assert(mapped == 1 && store.slots_allocated() == 64);
        assert(store.shrink_chunks(1) == 0);
        for (int e = 0; e < 64; ++e) f.equal(store, store.find(0, e));
    }
    std::puts("PASS mapped growth forbidden, visible slots unchanged");
    {
        available = 1ull << 30;
        // Native asynchronous disk prefetch and a simultaneous duplicate admission keep the original file data.
        FILE* file = std::tmpfile();
        assert(file != nullptr);
        for (int r = 0; r < 3; ++r) {
            assert(std::fseek(file, (long) f.geo.off[r], SEEK_SET) == 0);
            assert(std::fwrite(f.parts[r], 1, f.sizes[r], file) == f.sizes[r]);
        }
        assert(std::fflush(file) == 0);
        {
            ExpertStore store;
            auto c = f.cfg(1, available);
            assert(store.init(c, {f.geo}, missing, {fileno(file)}));
            const int e = 0;
            { std::lock_guard<std::mutex> lk(read_mu); gated_fd = fileno(file); }
            store.prefetch(0, e);
            {
                std::unique_lock<std::mutex> lk(read_mu);
                assert(read_cv.wait_for(lk, std::chrono::seconds(5), [] { return read_entered; }));
            }
            assert(store.find(0, 0) == -1);  // the pending read cannot have published a partial blob
            const auto queued = store.stats().bytes_read.load();
            assert(queued == (int64_t) (f.sizes[0] + f.sizes[1] + f.sizes[2]));
            assert(store.retain_free(0, 0, f.parts, f.sizes) == R::present);
            assert(store.retain_free(0, 1, f.parts, f.sizes) == R::no_room);
            { std::lock_guard<std::mutex> lk(read_mu); allow_read = true; }
            read_cv.notify_all();
            std::vector<char> ok;
            assert(store.ensure(0, &e, 1, ok) && ok[0]);
            f.equal(store, store.find(0, 0));
            assert(store.stats().bytes_read == queued && store.stats().fails == 0);
            store.release(0, &e, 1, ok);
        }
        std::fclose(file);
    }
    std::puts("PASS asynchronous prefetch is not replaced or reread");
}
