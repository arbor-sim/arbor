#pragma once

#include <string>
#include <vector>

#include <arbor/mechanism.hpp>

#include "backends/event.hpp"
#include "backends/multicore/multi_event_stream.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/shared_state.hpp"
#include "backends/multicore/diffusion_solver.hpp"
#include "backends/multicore/cable_solver.hpp"
#include "backends/multicore/threshold_watcher.hpp"
#include "execution_context.hpp"
#include "util/padded_alloc.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace multicore {

struct backend {
    static bool is_supported() { return true; }
    static std::string name() { return "cpu"; }

    using value_type = arb_value_type;
    using index_type = arb_index_type;
    using size_type  = arb_size_type;

    using array  = arb::multicore::array;
    using iarray = arb::multicore::iarray;

    static constexpr arb_backend_kind kind = arb_backend_kind_cpu;

    static util::range<const value_type*> host_view(const array& v) {
        return util::range_pointer_view(v);
    }

    static util::range<const index_type*> host_view(const iarray& v) {
        return util::range_pointer_view(v);
    }

    using cable_solver             = arb::multicore::cable_solver;
    using diffusion_solver         = arb::multicore::diffusion_solver;
    using threshold_watcher        = arb::multicore::threshold_watcher;
    using deliverable_event_stream = arb::multicore::deliverable_event_stream;
    using sample_event_stream      = arb::multicore::sample_event_stream;
    using shared_state             = arb::multicore::shared_state;
    using ion_state                = arb::multicore::ion_state;

    static threshold_watcher voltage_watcher(
        shared_state& state,
        const std::vector<index_type>& detector_cv,
        const std::vector<value_type>& thresholds,
        const execution_context& context)
    {
        return threshold_watcher(
            state.cv_to_intdom.data(),
            state.src_to_spike.data(),
            &state.time,
            &state.time_to,
            state.voltage.size(),
            detector_cv,
            thresholds,
            context);
    }

    static value_type* mechanism_field_data(arb::mechanism* mptr, const std::string& field);
};

} // namespace multicore
} // namespace arb
