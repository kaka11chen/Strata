// Compile with -I<patched-upstream>/src/dsv41 and --gc-sections: this exercises
// the actual native formatter without linking a model or loading CUDA.
#define STRATA_VERSION "store-diagnostics-test"
#define main strata_dsv41_unreferenced_main
#include "ds4_main.cpp"
#undef main

int main() {
    strata::ds4::Stats before, after;
    before.tokens = 100;
    before.store_hits = 1000;
    before.store_misses = 2000;
    before.store_bytes = 1024000;
    before.store_wait_ms = 1000.5;
    before.cpu_ms = 2000;
    before.store_evictions = 10;
    before.store_fails = 6;
    before.store_prefetched = 8;
    before.store_pf_used = 3;
    before.store_pf_wasted = 0;
    before.store_slots = 17;
    before.store_slots_max = 40;
    after = before;
    after.tokens += 2;
    after.store_hits += 6;
    after.store_misses += 2;
    after.store_bytes += 5000000;
    after.store_wait_ms += 500.25;
    after.cpu_ms += 700.5;
    after.store_evictions += 3;
    after.store_fails += 1;
    after.store_prefetched += 3;
    after.store_pf_used += 2;
    after.store_pf_wasted += 1;
    after.store_slots = 9;
    print_store_stats("prefill", before, after);
    print_store_stats("decode", after, after);
}
