#pragma once

#include <map>
#include <string>

#include <backends/event.hpp>
#include <backends/fvm_types.hpp>
#include <common_types.hpp>
#include <event_queue.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <memory/wrappers.hpp>
#include <util/meta.hpp>
#include <util/rangeutil.hpp>
#include <util/span.hpp>

#include "matrix_state.hpp"
#include "multi_event_stream.hpp"
#include "stimulus.hpp"
#include "threshold_watcher.hpp"

namespace nest {
namespace mc {
namespace multicore {

struct backend {
    static bool is_supported() {
        return true;
    }

    /// define the real and index types
    using value_type = fvm_value_type;
    using size_type  = fvm_size_type;

    /// define storage types
    using array  = memory::host_vector<value_type>;
    using iarray = memory::host_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array  = array;
    using host_iarray = iarray;

    using host_view   = view;
    using host_iview  = iview;

    using matrix_state = nest::mc::multicore::matrix_state<value_type, size_type>;

    // re-expose common backend event types
    using deliverable_event = nest::mc::deliverable_event;
    using target_handle = nest::mc::target_handle;

    using deliverable_event_stream = nest::mc::multicore::multi_event_stream<deliverable_event>;

    using sample_event_stream = nest::mc::multicore::multi_event_stream<sample_event>;

    //
    // mechanism infrastructure
    //
    using ion = mechanisms::ion<backend>;

    using mechanism = mechanisms::mechanism_ptr<backend>;

    using stimulus = mechanisms::multicore::stimulus<backend>;

    static mechanism make_mechanism(
        const std::string& name,
        size_type mech_id,
        const_iview vec_ci,
        const_view vec_t, const_view vec_t_to, const_view vec_dt,
        view vec_v, view vec_i,
        const std::vector<value_type>& weights,
        const std::vector<size_type>& node_indices)
    {
        if (!has_mechanism(name)) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        return mech_map_.find(name)->second(mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, array(weights), iarray(node_indices));
    }

    static bool has_mechanism(const std::string& name) {
        return mech_map_.count(name)>0;
    }

    static std::string name() {
        return "cpu";
    }

    /// threshold crossing logic
    /// used as part of spike detection back end
    using threshold_watcher =
        nest::mc::multicore::threshold_watcher<value_type, size_type>;


    // perform min/max reductions on 'array' type
    template <typename V>
    static std::pair<V, V> minmax_value(const memory::host_vector<V>& v) {
        return util::minmax_value(v);
    }

    // perform element-wise comparison on 'array' type against `t_test`.
    template <typename V>
    static bool any_time_before(const memory::host_vector<V>& t, V t_test) {
        return minmax_value(t).first<t_test;
    }

    static void update_time_to(array& time_to, const_view time, value_type dt, value_type tmax) {
        size_type ncell = util::size(time);
        for (size_type i = 0; i<ncell; ++i) {
            time_to[i] = std::min(time[i]+dt, tmax);
        }
    }

    // set the per-cell and per-compartment dt_ from time_to_ - time_.
    static void set_dt(array& dt_cell, array& dt_comp, const_view time_to, const_view time, const_iview cv_to_cell) {
        size_type ncell = util::size(dt_cell);
        size_type ncomp = util::size(dt_comp);

        for (size_type j = 0; j<ncell; ++j) {
            dt_cell[j] = time_to[j]-time[j];
        }

        for (size_type i = 0; i<ncomp; ++i) {
            dt_comp[i] = dt_cell[cv_to_cell[i]];
        }
    }

    // perform sampling as described by marked events in a sample_event_stream
    static void perform_marked_samples(value_type* store, const sample_event_stream& s) {
        for (size_type i = 0; i<s.size(); ++i) {
            for (const auto& ev: s.marked_events(i)) {
                store[ev.offset] = *ev.handle;
            }
        }
    }

private:
    using maker_type = mechanism (*)(value_type, const_iview, const_view, const_view, const_view, view, view, array&&, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism maker(value_type mech_id, const_iview vec_ci, const_view vec_t, const_view vec_t_to, const_view vec_dt, view vec_v, view vec_i, array&& weights, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<backend>>
            (mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, std::move(weights), std::move(node_indices));
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest
