#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <algorithms.hpp>
#include <util/pprintf.hpp>

namespace nest{ namespace mc{ namespace mechanisms{

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
        amplitude = amp;
        duration = dur;
        delay = del;
    }

    void nrn_current() override {
        if (amplitude.size() != size()) {
            throw std::domain_error("stimulus called with mismatched parameter size\n");
        }
        indexed_view_type vec_i(vec_i_, node_index_);
        int n = size();
        for(int i=0; i<n; ++i) {
            if (t>=delay[i] && t<(delay[i]+duration[i])) {
                // use subtraction because the electrod currents are specified
                // in terms of current into the compartment
                vec_i[i] -= amplitude[i];
            }
        }
    }

    value_type dt = 0;
    value_type t = 0;

    std::vector<value_type> amplitude;
    std::vector<value_type> duration;
    std::vector<value_type> delay;

    using base::vec_v_;
    using base::vec_i_;
    using base::node_index_;
};

}
}
} // namespaces

