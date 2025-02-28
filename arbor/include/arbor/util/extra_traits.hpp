#pragma once

namespace arb {

// Helper class, to be specialized in each cell header, mapping from Metadata to Value types
template <typename M>
struct probe_value_type_of {
    using meta_type = M;
    using type = void;
};

template <typename M>
using probe_value_type_of_t = probe_value_type_of<M>::type;

} // namespace arb
