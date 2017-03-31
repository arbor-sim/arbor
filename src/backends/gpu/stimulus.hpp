#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <algorithms.hpp>
#include <util/pprintf.hpp>

#include "intrinsics.hpp"

namespace nest{
namespace mc{
namespace mechanisms {
namespace gpu {

namespace kernels {
    template <typename T, typename I>
    __global__
    void stim_current(
        const T* delay, const T* duration, const T* amplitude,
        const I* node_index, int n, T t, T* current)
    {
        using value_type = T;
        using iarray = I;

        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        if (i<n) {
            if (t>=delay[i] && t<(delay[i]+duration[i])) {
                // use subtraction because the electrode currents are specified
                // in terms of current into the compartment
                cuda_atomic_add(current+node_index[i], -amplitude[i]);
            }
        }
    }
} // namespace kernels

template<class Backend>
class stimulus : public mechanism<Backend> {
public:
    using base = mechanism<Backend>;
    using value_type  = typename base::value_type;
    using size_type   = typename base::size_type;

    using array = typename base::array;
    using iarray  = typename base::iarray;
    using view   = typename base::view;
    using iview  = typename base::iview;
    using const_iview = typename base::const_iview;
    using indexed_view_type= typename base::indexed_view_type;
    using ion_type = typename base::ion_type;

    stimulus(view vec_v, view vec_i, iarray&& node_index):
        base(vec_v, vec_i, std::move(node_index))
    {}

    using base::size;

    std::size_t memory() const override {
        return 0;
    }

    void set_params(value_type t_, value_type dt_) override {
        t = t_;
        dt = dt_;
    }

    std::string name() const override {
        return "stimulus";
    }

    mechanismKind kind() const override {
        return mechanismKind::point;
    }

    bool uses_ion(ionKind k) const override {
        return false;
    }

    void set_ion(ionKind k, ion_type& i, std::vector<size_type>const& index) override {
        throw std::domain_error(
            nest::mc::util::pprintf("mechanism % does not support ion type\n", name()));
    }

    void nrn_init() override {}
    void nrn_state() override {}

    void net_receive(int i_, value_type weight) override {
        throw std::domain_error("stimulus mechanism should never receive an event\n");
    }

    void set_parameters(
        const std::vector<value_type>& amp,
        const std::vector<value_type>& dur,
        const std::vector<value_type>& del)
    {
        amplitude = memory::on_gpu(amp);
        duration = memory::on_gpu(dur);
        delay = memory::on_gpu(del);
    }

    void nrn_current() override {
        if (amplitude.size() != size()) {
            throw std::domain_error("stimulus called with mismatched parameter size\n");
        }

        // don't launch a kernel if there are no stimuli
        if (!size()) return;

        auto n = size();
        auto thread_dim = 192;
        dim3 dim_block(thread_dim);
        dim3 dim_grid((n+thread_dim-1)/thread_dim );

        kernels::stim_current<value_type, size_type><<<dim_grid, dim_block>>>(
            delay.data(), duration.data(), amplitude.data(),
            node_index_.data(), n, t,
            vec_i_.data()
        );

    }

    value_type dt = 0;
    value_type t = 0;

    array amplitude;
    array duration;
    array delay;

    using base::vec_v_;
    using base::vec_i_;
    using base::node_index_;
};

} // namespace gpu
} // namespace mechanisms
} // namespace mc
} // namespace nest
