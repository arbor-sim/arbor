#pragma once

#include <array>
#include <memory/memory.hpp>
#include <util/indirect.hpp>

namespace arb {

/*
  Ion channels have the following fields, whose label corresponds to that
  in NEURON. We give them more easily understood accessors.

    ---------------------------------------------------
    label   Ca      Na      K   name
    ---------------------------------------------------
    iX      ica     ina     ik  current
    eX      eca     ena     ek  reversal_potential
    Xi      cai     nai     ki  internal_concentration
    Xo      cao     nao     ko  external_concentration
    gX      gca     gna     gk  conductance
    ---------------------------------------------------
*/

/// enumerate the ion channel types
enum class ionKind {ca, na, k};

inline static
std::string to_string(ionKind k) {
    switch(k) {
        case ionKind::na : return "sodium";
        case ionKind::ca : return "calcium";
        case ionKind::k  : return "pottasium";
    }
    return "unkown";
}

/// a helper for iterting over the ion species
constexpr std::array<ionKind, 3> ion_kinds() {
    return {ionKind::ca, ionKind::na, ionKind::k};
}

/// storage for ion channel information in a cell group
template<typename Backend>
class ion {
public :
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

    ion(const std::vector<size_type>& idx) :
        node_index_{memory::make_const_view(idx)},
        iX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        eX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        Xi_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        Xo_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        valency(0),
        default_internal_concentration(0),
        default_external_concentration(0)
    {}

    std::size_t memory() const {
        return 4u*size() * sizeof(value_type)
               +  size() * sizeof(iarray)
               +  sizeof(ion);
    }

    view current() {
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

    void reset() {
        // The Nernst equation uses the assumption of nonzero concentrations:
        EXPECTS(default_internal_concentration> value_type(0));
        EXPECTS(default_external_concentration> value_type(0));
        memory::fill(iX_, 0);
        memory::fill(Xi_, default_internal_concentration);
        memory::fill(Xo_, default_external_concentration);
        nernst_reversal_potential(6.3+273.15); // TODO: use temperature specfied in model
    }

    /// Calculate the reversal potential for all compartments using Nernst equation
    /// temperature is in degrees Kelvin
    void nernst_reversal_potential(value_type temperature) {
        backend::nernst(valency, temperature, Xo_, Xi_, eX_);
    }

    const_iview node_index() const {
        return node_index_;
    }

    std::size_t size() const {
        return node_index_.size();
    }

private:
    iarray node_index_;
    array iX_;      // (nA) current
    array eX_;      // (mV) reversal potential
    array Xi_;      // (mM) internal concentration
    array Xo_;      // (mM) external concentration

public:
    int valency;    // valency of ionic species
    value_type default_internal_concentration; // (mM) default internal concentration
    value_type default_external_concentration; // (mM) default external concentration
};

} // namespace arb

