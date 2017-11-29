#pragma once

#include <cmath>
#include <limits>

#include <mechanism.hpp>
#include <algorithms.hpp>
#include <util/indirect.hpp>
#include <util/pprintf.hpp>

namespace arb{
namespace multicore{

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
        amplitude = amp;
        duration = dur;
        delay = del;
    }

    void set_weights(array&& w) override {
        EXPECTS(size()==w.size());
        weights.resize(size());
        std::copy(w.begin(), w.end(), weights.begin());
    }

    void nrn_current() override {
        if (amplitude.size() != size()) {
            throw std::domain_error("stimulus called with mismatched parameter size\n");
        }
        auto vec_t = util::indirect_view(util::indirect_view(vec_t_, vec_ci_), node_index_);
        auto vec_i = util::indirect_view(vec_i_, node_index_);
        size_type n = size();
        for (size_type i=0; i<n; ++i) {
            auto t = vec_t[i];
            if (t>=delay[i] && t<delay[i]+duration[i]) {
                // use subtraction because the electrod currents are specified
                // in terms of current into the compartment
                vec_i[i] -= weights[i]*amplitude[i];
            }
        }
    }

    std::vector<value_type> amplitude;
    std::vector<value_type> duration;
    std::vector<value_type> delay;
    std::vector<value_type> weights;

    using base::vec_ci_;
    using base::vec_t_;
    using base::vec_v_;
    using base::vec_i_;
    using base::node_index_;
};

} // namespace multicore
} // namespace arb
