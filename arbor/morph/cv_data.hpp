#pragma once

#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/morphology.hpp>

namespace arb {

struct cell_cv_data_impl {
    mcable_list cv_cables;                        // CV unbranched sections, partitioned by CV.
    std::vector<arb_index_type> cv_cables_divs;   // Partitions cv_cables by CV index.

    std::vector<arb_index_type> cv_parent;        // Index of CV parent or size_type(-1) for a cell root CV.
    std::vector<arb_index_type> cv_children;      // CV child indices, partitioned by CV, and then in order.
    std::vector<arb_index_type> cv_children_divs; // Paritions cv_children by CV index.

    cell_cv_data_impl() = default;
    cell_cv_data_impl(const cable_cell&, const locset&);
};

} // namespace arb
