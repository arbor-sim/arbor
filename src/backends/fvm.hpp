#pragma once

#include <map>
#include <vector>

#include <backends/event.hpp>
#include <backends/fvm_types.hpp>
#include <backends/multicore/fvm.hpp>
#include <backends/multicore/multi_event_stream.hpp>
#include <backends/multicore/matrix_state.hpp>
#include <backends/multicore/threshold_watcher.hpp>
#include <ion.hpp>
#include <memory/memory.hpp>

namespace arb {

// A null back end used as a placeholder for back ends that are not supported
// on the target platform.

struct null_backend {
    using value_type = fvm_value_type;
    using size_type  = fvm_size_type;

    using array = memory::host_vector<value_type>;
    using iarray = memory::host_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array = array;
    using host_view = array::view_type;

    using matrix_state = arb::multicore::matrix_state<value_type, size_type>;
    using deliverable_event_stream = arb::multicore::multi_event_stream<deliverable_event>;
    using sample_event_stream = arb::multicore::multi_event_stream<sample_event>;
    using threshold_watcher = arb::multicore::threshold_watcher<value_type, size_type>;

    struct shared_state {
        size_type n_cell = 0;
        size_type n_cv = 0;

        iarray cv_to_cell;
        array  time;
        array  time_to;
        array  dt;
        array  dt_comp;
        array  voltage;
        array  current_density;

        std::map<ionKind, ion<null_backend>> ion_data;
        deliverable_event_stream deliverable_events;
        sample_event_stream sample_events;

        shared_state() = default;
        shared_state(size_type, const std::vector<size_type>&) {}
        void add_ion(ionKind kind, ion<null_backend> ion) {}

        // debug interface only
        friend std::ostream& operator<<(std::ostream& o, const shared_state&) { return o; }
    };

    template <typename V>
    static std::pair<V, V> minmax_value(const memory::host_vector<V>&) { return {}; }

    static void update_time_to(array&, const_view, value_type, value_type) {}
    static void set_dt(array&, array&, const_view, const_view, const_iview) {}
    static void take_samples(const sample_event_stream::state&, const_view, array&, array&) {}
    static void nernst(int, value_type, const_view, const_view, view eX) {}
    static void init_concentration(view, view, const_view, const_view, value_type, value_type) {}

    static bool is_supported() { return false; }
    static std::string name() { return "null"; }
};

} // namespace arb

#ifdef ARB_HAVE_GPU
#include <backends/gpu/fvm.hpp>
#else
namespace arb {  namespace gpu {
    using backend = null_backend;
}} // namespace arb::gpu
#endif
