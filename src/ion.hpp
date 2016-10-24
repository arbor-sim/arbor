#pragma once

#include <memory/memory.hpp>

#include <indexed_view.hpp>

namespace nest {
namespace mc {
namespace mechanisms {

/*******************************************************************************

  ion channels have the following fields :

    ---------------------------------------------------
    label   Ca      Na      K   name
    ---------------------------------------------------
    iX      ica     ina     ik  current
    eX      eca     ena     ek  reversal_potential
    Xi      cai     nai     ki  internal_concentration
    Xo      cao     nao     ko  external_concentration
    gX      gca     gna     gk  conductance
    ---------------------------------------------------
*******************************************************************************/

/// let's enumerate the ion channel types
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

/// and a little helper to iterate over them
inline static
std::vector<ionKind> ion_kinds() {
    return {ionKind::ca, ionKind::na, ionKind::k};
}

/// storage for ion channel information in a cell group
template<typename MemoryTraits>
class ion : MemoryTraits{
public :
    using memory_traits = MemoryTraits;

    // expose tempalte parameters
    using typename memory_traits::value_type;
    using typename memory_traits::size_type;

    // define storage types
    using typename memory_traits::vector_type;
    using typename memory_traits::index_type;
    using typename memory_traits::view;
    using typename memory_traits::const_iview;

    using indexed_view_type = indexed_view<memory_traits>;

    ion() = default;

    ion(const_iview idx) :
        node_index_{idx},
        iX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        eX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        Xi_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()},
        Xo_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()}
    {}

    std::size_t memory() const {
        return 4u*size() * sizeof(value_type)
               +  size() * sizeof(index_type)
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

    index_type node_index_;
    vector_type iX_;
    vector_type eX_;
    vector_type Xi_;
    vector_type Xo_;
};

} // namespace mechanisms
} // namespace mc
} // namespace nest

