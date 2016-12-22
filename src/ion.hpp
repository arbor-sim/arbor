#pragma once

#include <array>
#include <memory/memory.hpp>
#include <indexed_view.hpp>

namespace nest {
namespace mc {
namespace mechanisms {

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

    using indexed_view_type = indexed_view<backend>;

    ion() = default;

    ion(const std::vector<size_type>& idx) :
        node_index_{memory::make_const_view(idx)},
        iX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        eX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        Xi_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        Xo_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()}
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

    const_iview node_index() const {
        return node_index_;
    }

    std::size_t size() const {
        return node_index_.size();
    }

private :

    iarray node_index_;
    array iX_;
    array eX_;
    array Xi_;
    array Xo_;
};

} // namespace mechanisms
} // namespace mc
} // namespace nest

