#pragma once

#include <arbor/arb_types.hpp>
#include <arbor/common_types.hpp>

namespace arb {

struct fvm_gap_junction {
    cell_lid_type local_idx; // Index relative to other gap junction sites on the cell.
    arb_size_type local_cv;  // CV index of the local gap junction site.
    arb_size_type peer_cv;   // CV index of the peer gap junction site.
    arb_value_type weight;   // unit-less local weight of the connection.

    constexpr bool operator==(const fvm_gap_junction&) const = default;
    constexpr auto operator<=>(const fvm_gap_junction& o) const {
        return std::tie(local_cv, peer_cv, local_idx, weight) <=> std::tie(o.local_cv, o.peer_cv, o.local_idx, o.weight);
    }
};

} // namespace arb
