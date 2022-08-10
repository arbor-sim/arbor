#pragma once

#include <map>
#include <string>

#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>

#include "memory/memory.hpp"
#include "util/rangeutil.hpp"

#include "backends/event.hpp"

#include "backends/gpu/gpu_store_types.hpp"
#include "backends/gpu/shared_state.hpp"

#include "threshold_watcher.hpp"

namespace arb {
namespace gpu {

struct backend {
    static bool is_supported() { return true; }
    static std::string name() { return "gpu"; }

    using value_type = arb_value_type;
    using index_type = arb_index_type;
    using size_type  = arb_size_type;

    using array  = arb::gpu::array;
    using iarray = arb::gpu::iarray;

    static constexpr arb_backend_kind kind = arb_backend_kind_gpu;

    static memory::host_vector<value_type> host_view(const array& v) {
        return memory::on_host(v);
    }

    static memory::host_vector<index_type> host_view(const iarray& v) {
        return memory::on_host(v);
    }

    using threshold_watcher        = arb::gpu::threshold_watcher;
    using cable_solver             = arb::gpu::matrix_state_fine<arb_value_type, arb_index_type>;
    using diffusion_solver         = arb::gpu::diffusion_state<arb_value_type, arb_index_type>;
    using deliverable_event_stream = arb::gpu::deliverable_event_stream;
    using sample_event_stream      = arb::gpu::sample_event_stream;

    using shared_state = arb::gpu::shared_state;
    using ion_state = arb::gpu::ion_state;

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
};

} // namespace gpu
} // namespace arb
