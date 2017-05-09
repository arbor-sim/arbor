#pragma once

#include <map>
#include <string>

#include <backends/event.hpp>
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

    using matrix_state = nest::mc::multicore::matrix_state<value_type, size_type>;
    using multi_event_stream = nest::mc::multicore::multi_event_stream;

    // re-expose common backend event types
    using deliverable_event = nest::mc::deliverable_event;
    using target_handle = nest::mc::target_handle;

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

    static void update_time_to(array& time_to, const_view time, value_type dt, value_type tmax) {
        size_type ncell = util::size(time);
        for (size_type i = 0; i<ncell; ++i) {
            time_to[i] = std::min(time[i]+dt, tmax);
        }
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
