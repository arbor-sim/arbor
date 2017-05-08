#pragma once

#include <map>
#include <string>

#include <common_types.hpp>
#include <event_queue.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <memory/wrappers.hpp>
#include <util/meta.hpp>
#include <util/rangeutil.hpp>
#include <util/span.hpp>

#include "matrix_state.hpp"
#include "stimulus.hpp"
#include "threshold_watcher.hpp"

namespace nest {
namespace mc {
namespace multicore {

struct backend {
    /// define the real and index types
    using value_type = double;
    using size_type  = nest::mc::cell_lid_type;

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

    /// matrix state
    using matrix_state =
        nest::mc::multicore::matrix_state<value_type, size_type>;

    // per-cell event queue
    // note: `cell_index` field is strictly speaking redundant: can look up via vec_ci.
    struct target_handle {
        size_type mech_id;    // mechanism type identifier (per cell group).
        size_type index;      // instance of the mechanism
        size_type cell_index; // which cell (acts as index into e.g. vec_t)

        target_handle() {}
        target_handle(size_type mech_id, size_type index, size_type cell_index):
            mech_id(mech_id), index(index), cell_index(cell_index) {}
    };

    struct deliverable_event {
        value_type time;
        size_type mech_id;
        size_type index;
        value_type weight;

        deliverable_event() {}
        deliverable_event(value_type time, target_handle handle, value_type weight):
            time(time), mech_id(handle.mech_id), index(handle.index), weight(weight) {}
    };

    using cell_event_queue = multi_event_stream<deliverable_event>;

    static void mark_events(const_view vec_t, cell_event_queue& events) {
        size_type ncell = util::size(vec_t);
        for (size_type c = 0; c<ncell; ++c) {
            events.mark_until_after(c, vec_t[c]);
        }
    }

    static void retire_events(value_type dt_max, value_type t_max, view vec_t, view vec_t_to, cell_event_queue& events) {
        size_type ncell = util::size(vec_t);
        for (size_type c = 0; c<ncell; ++c) {
            events.drop_marked_events(c);
            value_type max_t_to = std::min(vec_t[c]+dt_max, t_max);
            vec_t_to[c] = events.event_time_if_before(c, max_t_to);
        }
    }

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
        const_view vec_t, const_view vec_t_to,
        view vec_v, view vec_i,
        const std::vector<value_type>& weights,
        const std::vector<size_type>& node_indices)
    {
        if (!has_mechanism(name)) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        return mech_map_.find(name)->second(mech_id, vec_ci, vec_t, vec_t_to, vec_v, vec_i, array(weights), iarray(node_indices));
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
    static std::pair<value_type, value_type> minmax_value(const array& v) {
        return util::minmax_value(v);
    }

private:
    using maker_type = mechanism (*)(value_type, const_iview, const_view, const_view, view, view, array&&, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism maker(value_type mech_id, const_iview vec_ci, const_view vec_t, const_view vec_t_to, view vec_v, view vec_i, array&& weights, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<backend>>
            (mech_id, vec_ci, vec_t, vec_t_to, vec_v, vec_i, std::move(weights), std::move(node_indices));
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest
