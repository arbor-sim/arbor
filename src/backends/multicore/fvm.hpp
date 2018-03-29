#pragma once

#include <string>
#include <vector>

#include <backends/event.hpp>
#include <util/padded_alloc.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>

#include "matrix_state.hpp"
#include "multi_event_stream.hpp"
#include "multicore_common.hpp"
#include "shared_state.hpp"
#include "threshold_watcher.hpp"

namespace arb {
namespace multicore {

struct backend {
    static bool is_supported() { return true; }
    static std::string name() { return "cpu"; }

    using value_type = fvm_value_type;
    using size_type  = fvm_size_type;

    using array  = arb::multicore::array;
    using iarray = arb::multicore::iarray;

    static util::range<const value_type*> host_view(const array& v) {
        return util::range_pointer_view(v);
    }

    static util::range<const size_type*> host_view(const iarray& v) {
        return util::range_pointer_view(v);
    }

    using matrix_state = arb::multicore::matrix_state<value_type, size_type>;
    using threshold_watcher = arb::multicore::threshold_watcher;

    using deliverable_event_stream = arb::multicore::deliverable_event_stream;
    using sample_event_stream = arb::multicore::sample_event_stream;

    using shared_state = arb::multicore::shared_state;

    static threshold_watcher voltage_watcher(
        const shared_state& state,
        const std::vector<size_type>& cv,
        const std::vector<value_type>& thresholds)
    {
        return threshold_watcher(
            state.cv_to_cell.data(),
            state.time.data(),
            state.time_to.data(),
            state.voltage.data(),
            cv,
            thresholds);
    }
};

} // namespace multicore
} // namespace arb
