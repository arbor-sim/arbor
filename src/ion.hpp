#pragma once

#include <stdexcept>
#include <string>

#include <array>
#include <constants.hpp>
#include <memory/memory.hpp>
#include <util/indirect.hpp>

namespace arb {

/*
  Ion channels have the following fields, whose label corresponds to that
  in NEURON. We give them more easily understood accessors.

    ---------------------------------------------------
    label   Ca      Na      K   name
    ---------------------------------------------------
    iX      ica     ina     ik  current_density
    eX      eca     ena     ek  reversal_potential
    Xi      cai     nai     ki  internal_concentration
    Xo      cao     nao     ko  external_concentration
    gX      gca     gna     gk  conductance
    ---------------------------------------------------
*/

// Fixed set of ion species (to be generalized in the future):

enum class ionKind {ca, na, k};
inline std::string to_string(ionKind k) {
    switch (k) {
    case ionKind::ca: return "ca";
    case ionKind::na: return "na";
    case ionKind::k:  return "k";
    default: throw std::out_of_range("unknown ionKind");
    }
}

// Backend-specific ion state.

template<typename Backend>
class ion {
public:
    using backend = Backend;

    // expose tempalte parameters
    using value_type = typename backend::value_type;
    using size_type = typename backend::size_type;

    // define storage types
    using array = typename backend::array;
    using iarray = typename backend::iarray;
    using view = typename backend::view;
    using const_iview = typename backend::const_iview;

    ion() = default;

    ion(const std::vector<size_type>& idx):
        node_index_{memory::make_const_view(idx)},
        iX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        eX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        Xi_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        Xo_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        charge(0),
        default_int_concentration(0),
        default_ext_concentration(0)
    {}

    // Set the weights used when setting default concentration values in each CV.
    // The concentration of an ion species in a CV is a linear combination of
    // default concentration and contributions from mechanisms that update the
    // concentration. The weight is a value between 0 and 1 that represents the
    // proportion of the CV area for which the default value is to be used
    // (i.e. the proportion of the CV where the concentration is prescribed by a
    // mechanism).
    void set_weights(const std::vector<value_type>& win, const std::vector<value_type>& wout) {
        EXPECTS(win.size()  == size());
        EXPECTS(wout.size() == size());
        weight_Xi_ = memory::make_const_view(win);
        weight_Xo_ = memory::make_const_view(wout);
    }

    view current_density() {
        return iX_;
    }

    view reversal_potential() {
        return eX_;
    }

    view internal_concentration() {
        return Xi_;
    }

    view external_concentration() {
        return Xo_;
    }

    view internal_concentration_weights() {
        return weight_Xi_;
    }

    view external_concentration_weights() {
        return weight_Xo_;
    }

    void reset(double temperature_K) {
        // The Nernst equation uses the assumption of nonzero concentrations:
        EXPECTS(default_int_concentration > value_type(0));
        EXPECTS(default_ext_concentration > value_type(0));
        memory::fill(iX_, 0); // reset current
        init_concentration(); // reset internal and external concentrations
        nernst_reversal_potential(temperature_K);
    }

    // Calculate the reversal potential for all compartments using Nernst equation.
    // Temperature is in degrees Kelvin
    void nernst_reversal_potential(value_type temperature_K) {
        backend::nernst(charge, temperature_K, Xo_, Xi_, eX_);
    }

    void init_concentration() {
        backend::init_concentration(
            Xi_, Xo_, weight_Xi_, weight_Xo_,
            default_int_concentration, default_ext_concentration);
    }

    const_iview node_index() const {
        return node_index_;
    }

    std::size_t size() const {
        return node_index_.size();
    }

private:
    iarray node_index_;
    array iX_;          // (nA) current
    array eX_;          // (mV) reversal potential
    array Xi_;          // (mM) internal concentration
    array Xo_;          // (mM) external concentration
    array weight_Xi_;   // (1) concentration weight internal
    array weight_Xo_;   // (1) concentration weight external

public:
    int charge;    // charge of ionic species
    value_type default_int_concentration; // (mM) default internal concentration
    value_type default_ext_concentration; // (mM) default external concentration
};

} // namespace arb

