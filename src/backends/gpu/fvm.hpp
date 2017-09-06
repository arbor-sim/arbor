#pragma once

#include <map>
#include <string>

#include <backends/event.hpp>
#include <backends/fvm_types.hpp>
#include <common_types.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <util/rangeutil.hpp>

#include "kernels/time_ops.hpp"
#include "matrix_state_interleaved.hpp"
#include "matrix_state_flat.hpp"
#include "multi_event_stream.hpp"
#include "stimulus.hpp"
#include "threshold_watcher.hpp"

namespace nest {
namespace mc {
namespace gpu {

struct backend {
    static bool is_supported() {
        return true;
    }

    /// define the real and index types
    using value_type = fvm_value_type;
    using size_type  = fvm_size_type;

    /// define storage types
    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array  = typename memory::host_vector<value_type>;
    using host_iarray = typename memory::host_vector<size_type>;

    using host_view   = typename host_array::view_type;
    using host_iview  = typename host_iarray::const_view_type;

    static std::string name() {
        return "gpu";
    }

    // matrix back end implementation
    using matrix_state = matrix_state_interleaved<value_type, size_type>;
    using multi_event_stream = nest::mc::gpu::multi_event_stream;

    // re-expose common backend event types
    using deliverable_event = nest::mc::deliverable_event;
    using target_handle = nest::mc::target_handle;

    // mechanism infrastructure
    using ion = mechanisms::ion<backend>;

    using mechanism = mechanisms::mechanism_ptr<backend>;

    using stimulus = mechanisms::gpu::stimulus<backend>;

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

        return mech_map_.find(name)->
            second(mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, memory::make_const_view(weights), memory::make_const_view(node_indices));
    }

    static bool has_mechanism(const std::string& name) {
        return mech_map_.count(name)>0;
    }

    using threshold_watcher = nest::mc::gpu::threshold_watcher;

    // perform min/max reductions on 'array' type
    template <typename V>
    static std::pair<V, V> minmax_value(const memory::device_vector<V>& v) {
        // TODO: consider/test replacement with CUDA kernel (or generic reduction kernel).
        auto v_copy = memory::on_host(v);
        return util::minmax_value(v_copy);
    }

    // perform element-wise comparison on 'array' type against `t_test`.
    template <typename V>
    static bool any_time_before(const memory::device_vector<V>& t, V t_test) {
        // Note: benchmarking (on a P100) indicates that using the gpu::any_time_before
        // function is slower than the copy, unless we're running over ten thousands of
        // cells per cell group.
        //
        // Commenting out for now, but consider a size-dependent test or adaptive choice.

        // return gpu::any_time_before(t.size(), t.data(), t_test);

        auto v_copy = memory::on_host(t);
        return util::minmax_value(v_copy).first<t_test;
    }

    static void update_time_to(array& time_to, const_view time, value_type dt, value_type tmax) {
        nest::mc::gpu::update_time_to<value_type, size_type>(time_to.size(), time_to.data(), time.data(), dt, tmax);
    }

    // set the per-cell and per-compartment dt_ from time_to_ - time_.
    static void set_dt(array& dt_cell, array& dt_comp, const_view time_to, const_view time, const_iview cv_to_cell) {
        size_type ncell = util::size(dt_cell);
        size_type ncomp = util::size(dt_comp);

        nest::mc::gpu::set_dt<value_type, size_type>(ncell, ncomp, dt_cell.data(), dt_comp.data(), time_to.data(), time.data(), cv_to_cell.data());
    }

private:
    using maker_type = mechanism (*)(size_type, const_iview, const_view, const_view, const_view, view, view, array&&, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism maker(size_type mech_id, const_iview vec_ci, const_view vec_t, const_view vec_t_to, const_view vec_dt, view vec_v, view vec_i, array&& weights, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<backend>>
            (mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, std::move(weights), std::move(node_indices));
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest
