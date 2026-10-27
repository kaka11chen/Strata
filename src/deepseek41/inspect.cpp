#include "strata/deepseek41/artifact.hpp"

#include <charconv>
#include <cstdio>

namespace {
int parse_id(const char* value) {
    int id = -1;
    const char* end = value + std::strlen(value);
    const auto r = std::from_chars(value, end, id);
    if (r.ec != std::errc() || r.ptr != end || id < 0)
        throw std::invalid_argument("expected a nonnegative integer id");
    return id;
}
void usage() {
    std::fprintf(stderr, "usage: strata-deepseek41-inspect MODEL.gguf [--metadata-only | --layer N [--read-expert E]]\n"
                         "Checks core 1017 metadata and routed-expert byte layouts, not inference readiness.\n");
}
} // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        usage();
        return 0;
    }
    if (argc < 2) {
        usage();
        return 2;
    }
    bool metadata_only = false;
    int layer = -1, expert = -1;
    try {
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--metadata-only" && !metadata_only)
                metadata_only = true;
            else if (arg == "--layer" && layer < 0 && i + 1 < argc)
                layer = parse_id(argv[++i]);
            else if (arg == "--read-expert" && expert < 0 && i + 1 < argc)
                expert = parse_id(argv[++i]);
            else
                throw std::invalid_argument("unknown, repeated or incomplete argument: " + arg);
        }
        if ((metadata_only && (layer >= 0 || expert >= 0)) || (expert >= 0 && layer < 0))
            throw std::invalid_argument("--read-expert needs --layer; --metadata-only cannot select experts");
    } catch (const std::exception& e) {
        usage();
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    try {
        const auto model = strata::GgufModel::open(argv[1]);
        const auto g = strata::deepseek41::read_1017_geometry(model.meta());
        std::printf("1017 core profile: hidden=%d layers=%d experts=%d top_k=%d expert_width=%d shards=%zu\n", g.hidden,
                    g.layers, g.experts, g.top_k, g.expert_width, model.size());
        std::printf("Scope: core text metadata and routed experts only; full inference is not implemented.\n");
        if (metadata_only) {
            std::printf("Expert payloads: NOT CHECKED (--metadata-only).\n");
            return 0;
        }
        std::vector<int> layers;
        if (layer >= 0)
            layers.push_back(layer);
        const strata::deepseek41::ExpertIndex index(model, g, layers);
        std::printf("Expert layout: %zu/%d layers checked, %llu payload bytes%s\n", index.layer_count(), g.layers,
                    static_cast<unsigned long long>(index.total_bytes()),
                    layer >= 0 ? " (explicit partial check)" : "");
        if (layer >= 0) {
            const auto& s = g.sources.at(size_t(layer));
            std::printf("Layer %d sources: KV=%d index=%d candidates=%d\n", layer, s.kv, s.index, s.candidates);
        }
        if (expert >= 0) {
            std::vector<uint8_t> bytes(index.view(layer, expert).bytes());
            index.copy(layer, expert, bytes);
            uint64_t hash = 14695981039346656037ull;
            for (uint8_t byte : bytes) {
                hash ^= byte;
                hash *= 1099511628211ull;
            }
            std::printf("Read layer=%d expert=%d bytes=%zu fnv1a64=%016llx\n", layer, expert, bytes.size(),
                        static_cast<unsigned long long>(hash));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
