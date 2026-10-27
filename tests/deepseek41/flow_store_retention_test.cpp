// The same native parser used by each model; no model files or GPU needed.
#include "ds4_prefill_schedule.hpp"
#include <cassert>
#include <initializer_list>

int main() {
    using strata::ds4::flow_store_min_heat;
    static_assert(flow_store_min_heat() == 4.0f);
    static_assert(flow_store_min_heat("4") == 4.0f);
    static_assert(flow_store_min_heat("1") == 1.0f);
    for (const char* setting : {"", "0", "2", "-1", "1.0", "01", " 1", "1 ", "1x", "nan", "inf"})
        assert(flow_store_min_heat(setting) == 4.0f);
}
