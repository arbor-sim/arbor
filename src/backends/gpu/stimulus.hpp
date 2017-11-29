#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <algorithms.hpp>
#include "stim_current.hpp"
#include <util/pprintf.hpp>

namespace arb{
namespace gpu {

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
    using const_view = typename base::const_view;
    using const_iview = typename base::const_iview;
    using ion_type = typename base::ion_type;

    static constexpr size_type no_mech_id = (size_type)-1;

    stimulus(const_iview vec_ci, const_view vec_t, const_view vec_t_to, const_view vec_dt, view vec_v, view vec_i, iarray&& node_index):
        base(no_mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, std::move(node_index))
    {}

    using base::size;

    std::size_t memory() const override {
        return 0;
    }

    void set_params() override {}

    std::string name() const override {
        return "stimulus";
    }

    mechanismKind kind() const override {
        return mechanismKind::point;
    }

    typename base::ion_spec uses_ion(ionKind k) const override {
        return {false, false, false};
    }

    void set_ion(ionKind k, ion_type& i, std::vector<size_type>const& index) override {
        throw std::domain_error(
            arb::util::pprintf("mechanism % does not support ion type\n", name()));
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

    void set_weights(array&& w) override {
        EXPECTS(size()==w.size());
        weights = w;
    }

    void nrn_current() override {
        if (amplitude.size() != size()) {
            throw std::domain_error("stimulus called with mismatched parameter size\n");
        }

        // don't launch a kernel if there are no stimuli
        if (!size()) return;

        stim_current(delay.data(), duration.data(), amplitude.data(), weights.data(),
                     node_index_.data(), size(), vec_ci_.data(), vec_t_.data(),
                     vec_i_.data());

    }

    array amplitude;
    array duration;
    array delay;
    array weights;

    using base::vec_ci_;
    using base::vec_t_;
    using base::vec_v_;
    using base::vec_i_;
    using base::node_index_;
};

} // namespace gpu
} // namespace arb
