#pragma once

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
template<typename MemoryTraits>
class ion : public MemoryTraits {
public :
    using memory_traits = MemoryTraits;

    // expose tempalte parameters
    using typename memory_traits::value_type;
    using typename memory_traits::size_type;

    // define storage types
    using typename memory_traits::array;
    using typename memory_traits::iarray;
    using typename memory_traits::view;
    using typename memory_traits::const_iview;

    using indexed_view_type = indexed_view<memory_traits>;

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

