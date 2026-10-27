// Exercise the real native schedule state without a GPU. Native resident runs cover its model integration.
#include "ds4_prefill_schedule.hpp"
#include <cassert>
#include <cstdio>

using strata::ds4::MixedPrefillSchedule;
int main() {
    MixedPrefillSchedule disabled;
    for (int i = 0; i < 8; ++i) {
        disabled.begin_feed();
        assert(disabled.pick() == -1);
        disabled.complete();
    }
    MixedPrefillSchedule mixed(true);
    // Short inputs/all-logits feeds bypass the picker and must not consume the first flow.
    mixed.begin_feed();
    mixed.complete();
    mixed.begin_feed();
    assert(mixed.pick() == 1);
    // Failed/cancelled flow: no completion. An intervening bypassed feed must not commit that old pick.
    mixed.begin_feed();
    mixed.complete();
    mixed.begin_feed();
    assert(mixed.pick() == 1);
    mixed.complete();
    mixed.complete();  // completion cannot create a new pending choice
    // Fresh KV requests retain the model and its schedule. Subsequent failure/retry stays on windows.
    for (int i = 0; i < 4; ++i) {
        mixed.begin_feed();
        assert(mixed.pick() == 0);
        mixed.begin_feed();  // failed window attempt
        assert(mixed.pick() == 0);
        mixed.complete();
    }
    // New model lifetime starts with flow; another model's state cannot affect the existing one.
    MixedPrefillSchedule second(true);
    second.begin_feed();
    assert(second.pick() == 1);
    mixed.begin_feed();
    assert(mixed.pick() == 0);
    second.complete();
    second.begin_feed();
    assert(second.pick() == 0);
    std::puts("PASS: default delegation, bypassed feeds, failed/cancelled retry, success transition, repeated windows, model isolation");
}
