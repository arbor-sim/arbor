#pragma once

#include <vector/include/Vector.hpp>

#include "indexed_view.hpp"

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

[[gnu::unused]] static
std::string to_string(ionKind k)
{
    switch(k) {
        case ionKind::na : return "sodium";
        case ionKind::ca : return "calcium";
        case ionKind::k  : return "pottasium";
    }
    return "unkown";
}

/// and a little helper to iterate over them
[[gnu::unused]] static
std::vector<ionKind> ion_kinds()
{
    return {ionKind::ca, ionKind::na, ionKind::k};
}

/// storage for ion channel information in a cell group
template<typename T, typename I>
class ion {
public :
    // expose tempalte parameters
    using value_type  = T;
    using size_type   = I;

    // define storage types
    using vector_type      = memory::HostVector<value_type>;
    using index_type       = memory::HostVector<size_type>;
    using vector_view_type = typename vector_type::view_type;
    using index_view_type  = typename index_type::view_type;

    using indexed_view_type = indexed_view<value_type, size_type>;

    ion() = default;

    ion(index_view_type idx)
    :   node_index_{idx}
    ,   iX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()}
    ,   eX_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()}
    ,   Xi_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()}
    ,   Xo_{idx.size(), std::numeric_limits<value_type>::quiet_NaN()}
    { }

    std::size_t memory() const {
        return 4u*size() * sizeof(value_type)
               +  size() * sizeof(index_type)
               +  sizeof(ion);
    }

    vector_view_type current() {
        return iX_;
    }
    vector_view_type reversal_potential() {
        return eX_;
    }
    vector_view_type internal_concentration() {
        return Xi_;
    }
    vector_view_type external_concentration() {
        return Xo_;
    }

    index_type node_index() {
        return node_index_;
    }

    size_t size() const {
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

