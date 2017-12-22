#pragma once

#include <map>
#include <string>

#include <backends/event.hpp>
#include <backends/fvm_types.hpp>
#include <common_types.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <util/rangeutil.hpp>

#include "kernels/take_samples.hpp"
#include "matrix_state_interleaved.hpp"
#include "multi_event_stream.hpp"
#include "ions.hpp"
#include "stimulus.hpp"
#include "threshold_watcher.hpp"
#include "time_ops.hpp"

namespace arb {
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

    // dereference a probe handle
    static value_type dereference(probe_handle h) {
        memory::const_device_reference<value_type> v(h); // h is a device-side pointer
        return v;
    }

    // matrix back end implementation
    using matrix_state = matrix_state_interleaved<value_type, size_type>;

    // backend-specific multi event streams.
    using deliverable_event_stream = arb::gpu::multi_event_stream<deliverable_event>;
    using sample_event_stream = arb::gpu::multi_event_stream<sample_event>;

    // mechanism infrastructure
    using ion_type = ion<backend>;
    using stimulus = gpu::stimulus<backend>;

    using mechanism = arb::mechanism<backend>;
    using mechanism_ptr = std::unique_ptr<mechanism>;

    static mechanism_ptr make_mechanism(
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

    using threshold_watcher = arb::gpu::threshold_watcher;

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
        // Note: ubbench benchmarking (on a P100) indicates that copying the
        // time vectors to the host is faster than a device side
        // implementation unless we're running over ten thousands of cells per
        // cell group.

        auto v_copy = memory::on_host(t);
        return util::minmax_value(v_copy).first<t_test;
    }

    static void update_time_to(array& time_to, const_view time, value_type dt, value_type tmax) {
        arb::gpu::update_time_to(time_to.size(), time_to.data(), time.data(), dt, tmax);
    }

    // set the per-cell and per-compartment dt_ from time_to_ - time_.
    static void set_dt(array& dt_cell, array& dt_comp, const_view time_to, const_view time, const_iview cv_to_cell) {
        size_type ncell = util::size(dt_cell);
        size_type ncomp = util::size(dt_comp);

        arb::gpu::set_dt(
            ncell, ncomp, dt_cell.data(), dt_comp.data(), time_to.data(), time.data(), cv_to_cell.data());
    }

    // perform sampling as described by marked events in a sample_event_stream
    static void take_samples(
        const sample_event_stream::state& s,
        const_view time,
        array& sample_time,
        array& sample_value)
    {
        arb::gpu::take_samples(s, time.data(), sample_time.data(), sample_value.data());
    }

    // Calculate the reversal potential eX (mV) using Nernst equation
    // eX = RT/zF * ln(Xo/Xi)
    //      R: universal gas constant 8.3144598 J.K-1.mol-1
    //      T: temperature in Kelvin
    //      z: valency of species (K, Na: +1) (Ca: +2)
    //      F: Faraday's constant 96485.33289 C.mol-1
    //      Xo/Xi: ratio of out/in concentrations
    static void nernst(int valency, value_type temperature, const_view Xo, const_view Xi, view eX) {
        arb::gpu::nernst(eX.size(), valency, temperature, Xo.data(), Xi.data(), eX.data());
    }

    static void init_concentration(
            view Xi, view Xo,
            const_view weight_Xi, const_view weight_Xo,
            value_type c_int, value_type c_ext)
    {
        arb::gpu::init_concentration(Xi.size(), Xi.data(), Xo.data(), weight_Xi.data(), weight_Xo.data(), c_int, c_ext);
    }

private:
    using maker_type = mechanism_ptr (*)(size_type, const_iview, const_view, const_view, const_view, view, view, array&&, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism_ptr maker(size_type mech_id, const_iview vec_ci, const_view vec_t, const_view vec_t_to, const_view vec_dt, view vec_v, view vec_i, array&& weights, iarray&& node_indices) {
        return arb::make_mechanism<Mech<backend>>
            (mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, std::move(weights), std::move(node_indices));
    }
};

} // namespace gpu
} // namespace arb
