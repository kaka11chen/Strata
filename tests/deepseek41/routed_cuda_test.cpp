#include "strata/deepseek41/expert_source.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include "strata/deepseek41/routed_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "../core/gguf_fixture.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>

using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) {
    ++checks; failures += !ok;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void rejects(const std::string& name, const std::function<void()>& run) {
    try { run(); check(false, name); }
    catch (const std::exception& e) { check(true, name + ": " + e.what()); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct TempDir {
    fs::path path = fs::temp_directory_path() /
                    ("strata-ds41-route-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { fs::create_directories(path); }
    ~TempDir() { std::error_code e; fs::remove_all(path, e); }
};
float bf16(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
bool same_route(const Route& a, const Route& b) {
    return a.ids == b.ids && a.weights == b.weights;
}
std::vector<float> download(const float* p, int n, cudaStream_t stream) {
    std::vector<float> y(static_cast<size_t>(n));
    ck(cudaMemcpyAsync(y.data(), p, y.size() * 4, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream));
    return y;
}
// Independent FP64 router projection; route semantics have their separate CPU oracle test.
Route reference_route(const Geometry& g, std::span<const uint16_t> router, std::span<const float> bias,
                       const std::vector<float>& x) {
    std::array<float, 384> logits{};
    for (int e = 0; e < 384; ++e) {
        double sum = 0.;
        for (int i = 0; i < g.hidden; ++i)
            sum += double(kernels::f32_from_bf16(router[size_t(e) * g.hidden + i])) * bf16(x[size_t(i)]);
        logits[size_t(e)] = float(sum);
    }
    return route_top6(logits, bias);
}
std::vector<float> independent_sum(const Geometry& g, const ExpertIndex& index, const Route& route,
                                  const float* x, float* scratch, cudaStream_t stream) {
    std::vector<float> sum(size_t(g.hidden), 0.f);
    std::array<int, 6> order{};
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return route.ids[a] < route.ids[b]; });
    for (int k : order) {
        // Owned-weight single expert, no ExpertSource/cache/slot pointers. This
        // path already has an independent scalar matrix oracle in expert_cuda_test.
        CudaExpert expert(g, index.view(0, route.ids[k]));
        expert.forward(x, route.weights[k], scratch, stream);
        const auto y = download(scratch, g.hidden, stream);
        for (int i = 0; i < g.hidden; ++i) sum[size_t(i)] += y[size_t(i)];
    }
    return sum;
}
void run_cases(const Geometry& g, const ExpertIndex& index, std::span<const uint16_t> router,
               std::span<const float> bias, const std::vector<float>& input, const Route* golden = nullptr, const GgufModel* model = nullptr) {
    Mxfp4ExpertSource source(g, index);
    Stream stream;
    core::DeviceArena io(size_t(g.hidden) * 12 + 512);
    auto* x = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    auto* output = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    auto* temporary = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    ck(cudaMemcpyAsync(x, input.data(), input.size() * 4, cudaMemcpyHostToDevice, stream.value));
    CudaRoutedLayer layer(g, 0, router, bias, source, 6);
    const size_t allocated = layer.device_bytes(), staging = source.staging_bytes();
    const auto route = layer.forward(x, output, stream.value);
    const auto cold = download(output, g.hidden, stream.value);
    const auto reference = reference_route(g, router, bias, input);
    check(route.ids == reference.ids, "GPU router IDs match independent FP64 projection");
    float weight_error = 0.f;
    for (int k = 0; k < 6; ++k) weight_error = std::max(weight_error, std::abs(route.weights[k] - reference.weights[k]));
    check(weight_error <= 2e-6f, "GPU router weights match FP64 projection within 2e-6");
    if (golden) {
        check(route.ids == golden->ids, "real router IDs match hash-pinned official Gate.forward");
        float official_error = 0.f;
        for (int k = 0; k < 6; ++k)
            official_error = std::max(official_error, std::abs(route.weights[k] - golden->weights[k]));
        check(official_error <= 2e-6f, "real route weights match official Gate.forward within 2e-6");
        std::printf("official route max error %.9g; IDs", double(official_error));
        for (int id : route.ids) std::printf(" %d", id);
        std::printf("\n");
    }
    const auto initial = layer.stats();
    check(initial.misses == 6 && initial.hits == 0 && initial.evictions == 0 && source.reads() == 6,
          "cold top-6 reads and uploads exactly six experts");
    check(initial.expert_h2d_bytes == 6 * nvfp4_expert_bytes(g) && initial.router_d2h_bytes == 1536,
          "cold transfer byte accounting");
    const auto warm_route = layer.forward(x, output, stream.value);
    const auto warm = download(output, g.hidden, stream.value);
    check(same_route(route, warm_route) && same(cold, warm), "warm cache has bitwise identical route and output");
    check(layer.stats().hits == 6 && layer.stats().misses == 6 && source.reads() == 6 &&
          layer.stats().expert_h2d_bytes == initial.expert_h2d_bytes, "warm top-6 moves zero expert bytes");
    check(same(cold, independent_sum(g, index, route, x, temporary, stream.value)),
          "routed output equals independent sum of owned-weight GPU experts");
    check(std::all_of(cold.begin(), cold.end(), [](float v) { return std::isfinite(v); }) &&
          std::any_of(cold.begin(), cold.end(), [](float v) { return v != 0.f; }), "finite nonzero routed output");
    // Change input in place: the cache holds weights, never stale activations/results.
    if (!golden) {
        auto changed = input;
        for (float& v : changed) v = -v;
        ck(cudaMemcpyAsync(x, changed.data(), changed.size() * 4, cudaMemcpyHostToDevice, stream.value));
        const auto changed_route = layer.forward(x, output, stream.value);
        check(changed_route.ids != route.ids, "changed token selects a different original top-6");
        check(changed_route.ids == reference_route(g, router, bias, changed).ids,
              "cache residency does not alter a changed token's route");
        check(same(download(output, g.hidden, stream.value),
                   independent_sum(g, index, changed_route, x, temporary, stream.value)),
              "changed input at same address uses fresh activations");
        ck(cudaMemcpyAsync(x, input.data(), input.size() * 4, cudaMemcpyHostToDevice, stream.value));
        check(same_route(route, layer.forward(x, output, stream.value)) &&
              same(cold, download(output, g.hidden, stream.value)), "evicted original route reloads exactly");
    }
    // A cache smaller than top-k forces replacement inside the same token.
    Mxfp4ExpertSource small_source(g, index);
    CudaRoutedLayer small(g, 0, router, bias, small_source, 2);
    for (int iteration = 0; iteration < 3; ++iteration) {
        const auto evicted_route = small.forward(x, output, stream.value);
        check(same_route(route, evicted_route) && same(cold, download(output, g.hidden, stream.value)),
              "two-slot eviction/reload output is bitwise identical " + std::to_string(iteration));
    }
    check(small.stats().misses == 18 && small.stats().evictions == 16 && small.stats().hits == 0 &&
          small_source.reads() == 18, "three two-slot passes account for every LRU eviction");
    check(small.stats().expert_h2d_bytes == 18 * nvfp4_expert_bytes(g), "eviction transfer bytes stay explicit");
    // A global host tier can serve GPU misses without re-reading/reconverting.
    Mxfp4ExpertSource backing(g,index);
    const uint64_t host_budget=size_t(g.layers)*384*sizeof(int32_t)+6*(nvfp4_expert_bytes(g)+3*sizeof(int32_t));
    CachedExpertSource cached(g,backing,host_budget);
    CudaRoutedLayer tiered(g,0,router,bias,cached,2);
    for(int iteration=0;iteration<3;++iteration){
        check(same_route(route,tiered.forward(x,output,stream.value)) && same(cold,download(output,g.hidden,stream.value)),
              "bounded CPU cache preserves routed GPU output "+std::to_string(iteration));
    }
    const auto hs=cached.stats();
    check(hs.misses==6 && hs.hits==12 && !hs.evictions && backing.reads()==6,
          "host hits avoid twelve upstream conversions across GPU evictions");
    check(tiered.stats().misses==18 && cached.allocated_bytes()==host_budget && cached.resident_bytes()==6*nvfp4_expert_bytes(g),
          "CPU residency/accounting is independent of the two-slot GPU cache");
    std::vector<uint8_t> retained(nvfp4_expert_bytes(g)),expected(retained.size());
    pack_nvfp4_expert(g,index.view(0,route.ids[0]),expected);
    check(cached.copy_blob(0,route.ids[0],retained.data()) && retained==expected,"copy_blob returns caller-owned exact weights");
    rejects("evictable host cache cannot promise lifetime-stable pointers",[&]{cached.blob_stable(0,route.ids[0]);});
    rejects("host cache validates original IDs before upstream access",[&]{cached.blob(0,int64_t(1)<<32);});
    check(!cached.copy_blob(0,route.ids[0],nullptr),"null copy destination rejected without reading");
    const auto* cached_bytes=cached.blob(0,route.ids[0]);
    ck(cudaMemcpyAsync(temporary,cached_bytes,64,cudaMemcpyHostToDevice,stream.value));cached.note_async_read(cached_bytes,stream.value);
    std::array<uint8_t,64> copied{};ck(cudaMemcpy(copied.data(),temporary,64,cudaMemcpyDeviceToHost));
    check(std::memcmp(copied.data(),expected.data(),64)==0,"noted asynchronous host cache read completes with exact bytes");
    // Exercise the common disk source through the existing CPU and GPU tiers.
    if (!model) throw std::logic_error("Indexed source test needs its GGUF metadata");
    ExpertFileIndex files(g); files.add_gguf_layer(*model,0);
    IndexedExpertSource disk(std::move(files));
    CachedExpertSource disk_cache(g,disk,host_budget);
    CudaRoutedLayer disk_layer(g,0,router,bias,disk_cache,2);
    for(int iteration=0;iteration<3;++iteration){
        check(same_route(route,disk_layer.forward(x,output,stream.value)) && same(cold,download(output,g.hidden,stream.value)),
              "indexed disk source preserves routed GPU output "+std::to_string(iteration));
    }
    const auto disk_hits=disk_cache.stats();const auto disk_io=disk.io_stats();
    check(disk_hits.misses==6 && disk_hits.hits==12 && !disk_hits.evictions && disk_io.converted==6 && disk_io.reads==18,
          "indexed source performs no disk reads on CPU cache hits");
    auto bad_ids=route.ids;bad_ids.back()=384;
    rejects("indexed source validates the entire top-6 before reading",[&]{disk.begin_layer(0,bad_ids.data(),6);});
    check(disk.reads()==6,"invalid route leaves source I/O untouched");
    rejects("indexed source cannot promise lifetime-stable storage",[&]{disk.blob_stable(0,route.ids[0]);});
    check(disk.copy_blob(0,route.ids[0],retained.data()) && retained==expected,
          "indexed copy returns exact caller-owned native weights");
    const auto* disk_bytes=disk.blob(0,route.ids[0]);
    ck(cudaMemcpyAsync(temporary,disk_bytes,64,cudaMemcpyHostToDevice,stream.value));disk.note_async_read(disk_bytes,stream.value);
    disk.blob(0,route.ids[1]);ck(cudaMemcpy(copied.data(),temporary,64,cudaMemcpyDeviceToHost));
    check(std::memcmp(copied.data(),expected.data(),64)==0,"indexed source async read completes before staging reuse");
    auto ids = route.ids;
    std::sort(ids.begin(), ids.end());
    bool residency_ok = true;
    for (int k = 0; k < 6; ++k) residency_ok = residency_ok && (small.slot_of(ids[k]) >= 0) == (k >= 4);
    check(residency_ok && small.resident() == 2, "residency table forgets evicted experts");
    check(layer.device_bytes() == allocated && source.staging_bytes() == staging &&
          small.cache_bytes() == 2 * nvfp4_expert_bytes(g) && small_source.staging_bytes() == staging,
          "fixed GPU allocation and one host staging blob across reloads");
    std::printf("memory: six-slot cache=%zu device=%zu; two-slot cache=%zu device=%zu; host staging=%zu B\n",
                layer.cache_bytes(), layer.device_bytes(), small.cache_bytes(), small.device_bytes(), staging);
    std::printf("transfers: cold=%llu warm=0 three two-slot passes=%llu B (expert H2D)\n",
                (unsigned long long)initial.expert_h2d_bytes, (unsigned long long)small.stats().expert_h2d_bytes);
    rejects("aliased input/output rejected", [&] { layer.forward(x, x, stream.value); });
    rejects("null stream rejected", [&] { layer.forward(x, output, nullptr); });
    rejects("zero cache slots rejected", [&] { CudaRoutedLayer bad(g, 0, router, bias, source, 0); });
    rejects("transient source cannot promise stable storage", [&] { source.blob_stable(0, route.ids[0]); });
    rejects("source rejects invalid expert without narrowing", [&] { source.blob(0, int64_t(1) << 32); });
    CudaExpert scratch(g);
    rejects("scratch-only expert rejects owned forward", [&] { scratch.forward(x, 1.f, output, stream.value); });
    rejects("scratch-only expert checks blob size", [&] {
        scratch.forward_from_blob(reinterpret_cast<uint8_t*>(x), 4, {1.f,1.f,1.f}, x, 1.f, output, stream.value);
    });
}

void synthetic() {
    TempDir tmp;
    Geometry g;
    g.layers = 1; g.hidden = g.expert_width = 64; g.experts = 384; g.top_k = 6;
    std::vector<fixture::Tensor> tensors;
    for (const char* role : {"gate", "up", "down"})
        tensors.push_back({std::string("blk.0.ffn_") + role + "_exps.weight", {64, 64, 384}, 39, uint8_t(tensors.size() + 1)});
    const auto path = tmp.path / "synthetic.gguf";
    const auto written = fixture::write(path, {fixture::str("general.architecture", "deepseek41-routed-fixture"),
        fixture::str("strata.deepseek41.revision",std::string(checkpoint_1017_revision))}, tensors);
    {
        // Keep the fixture's independent code patterns, replace only scales with
        // representable powers of two; each original expert retains distinct bytes.
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        for (size_t role = 0; role < 3; ++role) {
            std::vector<uint8_t> bytes(64 * 64 / 32 * 17 * 384);
            for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = fixture::pattern(uint8_t(role + 1), i);
            for (size_t i = 0; i < bytes.size(); i += 17) bytes[i] = uint8_t(120 + (i / 17) % 3);
            // Alternate ordinary/global-half experts. Positive/negative input
            // routes exercise all three tail positions across hits and eviction.
            const size_t part=64*64/32*17;
            for(size_t expert=0;expert<12;expert+=2)if(expert%3==role){
                bytes[expert*part]=117;bytes[expert*part+1]=0x31;
            }
            f.seekp(std::streamoff(written.data_start + written.offsets[role]));
            f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        }
    }
    const auto model = GgufModel::open(path.string());
    const ExpertIndex index(model, g);
    std::vector<uint16_t> router(384 * 64, 0);
    std::array<float, 384> bias{};
    bias.fill(-10.f);
    for (int e = 0; e < 12; ++e) {
        router[size_t(e) * 64] = kernels::bf16_from_f32(float(e - 6) * .25f);
        bias[size_t(e)] = 0.f;
    }
    std::vector<float> input(64);
    for (int i = 0; i < 64; ++i) input[size_t(i)] = bf16(.2f + .1f * std::sin(float(i)));
    run_cases(g, index, router, bias, input, nullptr, &model);
    Mxfp4ExpertSource source(g, index);
    std::vector<uint8_t> expected(nvfp4_expert_bytes(g)), copied(expected.size());
    pack_nvfp4_expert(g, index.view(0, 4), expected);
    check(source.copy_blob(0, 4, copied.data()) && copied == expected,
          "source copy_blob uses DeepSeek dimensions instead of Qwen's global layout");
    const auto* first = source.blob(0, 4);
    const auto* second = source.blob(0, 8);
    pack_nvfp4_expert(g, index.view(0, 8), expected);
    check(first == second && std::memcmp(second, expected.data(), expected.size()) == 0,
          "source reuses one staging buffer with correct new expert bytes");
    // Honor the existing asynchronous source lifetime protocol.
    {
        Stream copy_stream;
        core::DeviceArena device(expected.size());
        void* destination = device.alloc(expected.size());
        ck(cudaMemcpyAsync(destination, second, expected.size(), cudaMemcpyHostToDevice, copy_stream.value));
        source.note_async_read(second, copy_stream.value);
        source.blob(0, 4);
        ck(cudaMemcpy(copied.data(), destination, copied.size(), cudaMemcpyDeviceToHost));
        check(copied == expected, "source async-read protocol protects recycled staging bytes");
    }
    // A missing selected expert must not expose a falsely resident slot.
    struct MissingSource : core::ExpertSource {
        const uint8_t* blob(int64_t, int64_t) override { return nullptr; }
    } missing;
    Stream stream;
    core::DeviceArena io(512);
    auto* x = static_cast<float*>(io.alloc(256));
    auto* y = static_cast<float*>(io.alloc(256));
    ck(cudaMemcpy(x, input.data(), 256, cudaMemcpyHostToDevice));
    struct InvalidTail : core::ExpertSource {
        Mxfp4ExpertSource& source;std::vector<uint8_t> bytes;
        InvalidTail(Mxfp4ExpertSource& s,size_t n):source(s),bytes(n){}
        const uint8_t* blob(int64_t l,int64_t e) override {
            const auto* p=source.blob(l,e);std::memcpy(bytes.data(),p,bytes.size());
            const std::array<float,4> tail{1.f,0.f,1.f,0.f};
            std::memcpy(bytes.data()+bytes.size()-nvfp4_scale_tail_bytes,tail.data(),nvfp4_scale_tail_bytes);
            return bytes.data();
        }
    } invalid_tail(source,nvfp4_expert_bytes(g));
    CudaRoutedLayer invalid(g,0,router,bias,invalid_tail,2);
    rejects("invalid scale tail rejected before cache admission",[&]{invalid.forward(x,y,stream.value);});
    check(invalid.resident()==0 && invalid.stats().misses==0,"invalid scale metadata never publishes residency");
    rejects("invalid scale execution poisons the instance",[&]{invalid.forward(x,y,stream.value);});
    CudaRoutedLayer broken(g, 0, router, bias, missing, 2);
    rejects("missing selected expert is an error, never rerouted", [&] { broken.forward(x, y, stream.value); });
    check(broken.resident() == 0, "failed fill publishes no residency");
    rejects("execution failure makes the instance unusable", [&] { broken.forward(x, y, stream.value); });
    struct FailsThird : core::ExpertSource {
        Mxfp4ExpertSource& source;
        int calls = 0;
        explicit FailsThird(Mxfp4ExpertSource& s) : source(s) {}
        const uint8_t* blob(int64_t l, int64_t e) override { return ++calls == 3 ? nullptr : source.blob(l, e); }
    } fails_third(source);
    CudaRoutedLayer partial(g, 0, router, bias, fails_third, 2);
    rejects("source failure after two GPU experts propagates", [&] { partial.forward(x, y, stream.value); });
    check(partial.resident() == 2 && partial.stats().misses == 2 && partial.stats().evictions == 0,
          "partially executed token publishes only successful fills");
    rejects("partial failure cannot silently reuse previous residency", [&] { partial.forward(x, y, stream.value); });
}

template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, uint32_t type, size_t n) {
    size_t shard = 0;
    const auto* t = model.find(name, &shard);
    if (!t || t->type != type || !model.in_bounds(*t, shard) || tensor_payload_bytes(*t) != n * sizeof(T))
        throw std::runtime_error("Bad routed fixture tensor " + name);
    std::vector<T> out(n);
    std::memcpy(out.data(), model.shard(shard).tensor_data(*t), n * sizeof(T));
    return out;
}
void real(const char* path) {
    const auto model = GgufModel::open(path);
    const auto* arch = model.meta().get("general.architecture");
    const auto* ids = model.meta().get("strata.deepseek41.selected_ids");
    if (!arch || arch->s != "deepseek41-routed-fixture" || !ids || ids->items.size() != 6)
        throw std::runtime_error("Expected a sparse routed fixture with official route metadata");
    Geometry g;
    g.layers = 40; g.hidden = 5120; g.expert_width = 2304; g.experts = 384; g.top_k = 6;
    const ExpertIndex index(model, g, {0});
    const auto router = tensor<uint16_t>(model, "router.weight", 30, 5120 * 384);
    const auto bias = tensor<float>(model, "router.bias", 0, 384);
    const auto input = tensor<float>(model, "input", 0, 5120);
    const auto weights = tensor<float>(model, "expected.weights", 0, 6);
    Route golden;
    for (int k = 0; k < 6; ++k) { golden.ids[k] = int32_t(ids->items[size_t(k)].u); golden.weights[k] = weights[size_t(k)]; }
    std::printf("Real 1017 layer-0 router and six experts, seeded BF16 input (not a model token trace)\n");
    run_cases(g, index, router, bias, input, &golden, &model);
    Mxfp4ExpertSource source(g, index);
    rejects("unpopulated sparse expert is rejected, not a zero-weight substitute", [&] { source.blob(0, 0); });
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 1 && !(argc == 3 && std::string(argv[1]) == "--real-routed"))
            throw std::runtime_error("usage: deepseek41_routed_cuda_test [--real-routed FIXTURE.gguf]");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        synthetic();
        if (argc == 3) real(argv[2]);
        std::printf("%d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
