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


#ifdef ARB_HAVE_GPU_FINE_MATRIX
    #include "matrix_state_fine.hpp"
#else
    #include "matrix_state_interleaved.hpp"
#endif

namespace arb {
namespace gpu {

struct backend {
    static bool is_supported() { return true; }
    static std::string name() { return "gpu"; }

    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type  = fvm_size_type;

    using array  = arb::gpu::array;
    using iarray = arb::gpu::iarray;

    static memory::host_vector<value_type> host_view(const array& v) {
        return memory::on_host(v);
    }

    static memory::host_vector<index_type> host_view(const iarray& v) {
        return memory::on_host(v);
    }

#ifdef ARB_HAVE_GPU_FINE_MATRIX
    using matrix_state = arb::gpu::matrix_state_fine<value_type, index_type>;
#else
    using matrix_state = arb::gpu::matrix_state_interleaved<value_type, index_type>;
#endif
    using threshold_watcher = arb::gpu::threshold_watcher;

    using deliverable_event_stream = arb::gpu::deliverable_event_stream;
    using sample_event_stream = arb::gpu::sample_event_stream;

    using shared_state = arb::gpu::shared_state;

    static threshold_watcher voltage_watcher(
        const shared_state& state,
        const std::vector<index_type>& cv,
        const std::vector<value_type>& thresholds,
        const execution_context& context)
    {
        return threshold_watcher(
            state.cv_to_intdom.data(),
            state.time.data(),
            state.time_to.data(),
            state.voltage.data(),
            cv,
            thresholds,
            context);
    }
};

} // namespace gpu
} // namespace arb
