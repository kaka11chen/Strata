// Test the real native first-feed state. Existing model callers own execution/cancellation.
#include "ds4_prefill_schedule.hpp"
#include <cassert>
#include <cstdio>
#include <initializer_list>

using strata::ds4::FirstWindowPrefill;
int main() {
    FirstWindowPrefill disabled;
    for (int n : {14, 16, 17, 257}) {
        disabled.begin_feed();
        assert(!disabled.pick(n, 16));
        disabled.complete();
    }
    for (int n : {14, 16, 17, 257}) {
        FirstWindowPrefill state(true);
        state.begin_feed();
        assert(state.pick(n, 16) == (n <= 16));
        assert(state.pick(n, 16) == (n <= 16));  // selection itself never commits
        state.complete();
        state.complete();
        state.begin_feed();  // new KV request keeps the same model-lifetime state
        assert(!state.pick(14, 16));
        state.complete();
    }
    for (int failed_n : {14, 17, 257}) {
        FirstWindowPrefill state(true);
        state.begin_feed();
        (void) state.pick(failed_n, 16);  // failed/cancelled: no complete()
        state.begin_feed();             // an intervening single-token/all-logits bypass
        state.complete();
        state.begin_feed();
        assert(state.pick(16, 16));
        state.complete();
        state.begin_feed();
        assert(!state.pick(16, 16));
    }
    FirstWindowPrefill first(true), second(true);
    first.begin_feed();
    assert(!first.pick(257, 16));  // a successful large first feed consumes the opportunity
    first.complete();
    first.begin_feed();
    assert(!first.pick(14, 16));
    second.begin_feed();
    assert(second.pick(14, 16));  // new model remains independent
    FirstWindowPrefill caller_limit(true);
    caller_limit.begin_feed();
    assert(!caller_limit.pick(14, 8));  // bound comes from the caller's native window size
    std::puts("PASS: disabled delegation,14/16/17/257 boundaries,success-only transition,bypass/cancel retry,large-first consumption,model isolation,caller window");
}
