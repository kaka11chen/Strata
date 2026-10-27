#include "strata/deepseek41/text_io_weights.hpp"
#include "../core/gguf_fixture.hpp"
#include <chrono>
#include <cstdio>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
int checks = 0, failures = 0;
void check(bool ok, const char* name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); }
void rejects(const char* name, const std::function<void()>& f) {
    try { f(); check(false, name); } catch (const std::exception&) { check(true, name); }
}
int main() {
    const auto dir = fs::temp_directory_path() / ("strata-text-io-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try {
        const std::vector<fixture::Kv> metadata{fixture::str("strata.deepseek41.text_io_profile", text_io_profile),
                                               fixture::str("strata.deepseek41.model_revision", text_io_revision)};
        const std::vector<fixture::Tensor> tensors{{"token_embd.weight", {64, 19}, 30, 1},
                                                  {"output_norm.weight", {64}, 30, 2}, {"output.weight", {64, 19}, 30, 3}};
        auto open = [&](const std::vector<fixture::Kv>& meta, const std::vector<fixture::Tensor>& ts, uint64_t cut = 0) {
            const auto path = dir / (std::to_string(checks) + ".gguf");
            fixture::write(path, meta, ts, cut); return GgufModel::open(path.string());
        };
        const auto model = open(metadata, tensors);
        const auto w = read_text_io_weights(model, 64, 19);
        check(w.hidden == 64 && w.vocab_size == 19 && w.embedding.size() == 1216 && w.head.size() == 1216 && w.norm.size() == 64,
              "typed BF16 views retain distinct embedding, norm and output shapes");
        size_t part = 0;
        for (const auto view : {w.embedding, w.norm, w.head}) {
            bool exact = true; const auto* raw = reinterpret_cast<const uint8_t*>(view.data());
            for (size_t i = 0; i < view.size_bytes(); ++i) exact &= raw[i] == fixture::pattern(uint8_t(part + 1), i);
            ++part; check(exact, "original BF16 bytes unchanged");
        }
        check(w.embedding.data() != w.head.data(), "untied embedding and head remain separate");
        for (size_t i = 0; i < metadata.size(); ++i) {
            rejects("missing pinned identity rejected", [&] { auto m = metadata; m.erase(m.begin() + std::ptrdiff_t(i)); read_text_io_weights(open(m, tensors), 64, 19); });
            rejects("wrong pinned identity rejected", [&] { auto m = metadata; m[i].s = "other"; read_text_io_weights(open(m, tensors), 64, 19); });
            rejects("wrong identity type rejected", [&] { auto m = metadata; m[i] = fixture::u32(m[i].key, 1); read_text_io_weights(open(m, tensors), 64, 19); });
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            rejects("missing original tensor rejected", [&] { auto t = tensors; t.erase(t.begin() + std::ptrdiff_t(i)); read_text_io_weights(open(metadata, t), 64, 19); });
            rejects("non-BF16 tensor rejected", [&] { auto t = tensors; t[i].type = 0; read_text_io_weights(open(metadata, t), 64, 19); });
            rejects("wrong tensor shape rejected", [&] { auto t = tensors; t[i].shape[0] = 32; read_text_io_weights(open(metadata, t), 64, 19); });
        }
        rejects("transposed embedding rejected", [&] { auto t = tensors; t[0].shape = {19, 64}; read_text_io_weights(open(metadata, t), 64, 19); });
        rejects("transposed output head rejected", [&] { auto t = tensors; t[2].shape = {19, 64}; read_text_io_weights(open(metadata, t), 64, 19); });
        rejects("truncated head rejected", [&] { read_text_io_weights(open(metadata, tensors, 1), 64, 19); });
        for (int hidden : {0, -32, 33, 5152}) rejects("invalid hidden width rejected", [&] { read_text_io_weights(model, hidden, 19); });
        for (int vocab : {0, -1, 129281}) rejects("invalid vocabulary rejected", [&] { read_text_io_weights(model, 64, vocab); });
        rejects("tiny fixture is not a full checkpoint", [&] { read_text_io_weights(model); });
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
