#pragma once

#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {
class cv_geometry;

// Stores info about the CV geometry of a discretized cable-cell
class cell_cv_geometry {
public:
    auto cables(fvm_size_type cv_index) const;   // Returns mcables comprising the CV at a given index.
    auto children(fvm_size_type cv_index) const; // Returns the CV indices of the children of a given CV index.
    auto parent(fvm_size_type cv_index) const;   // Returns the CV index of the parent of a given CV index.
    fvm_size_type num_cv() const;                // Returns total number of CVs.

private:
    std::vector<mcable> cv_cables;                // CV unbranched sections, partitioned by CV.
    std::vector<fvm_index_type> cv_cables_divs;   // Partitions cv_cables by CV index.

    std::vector<fvm_index_type> cv_parent;        // Index of CV parent or size_type(-1) for a cell root CV.
    std::vector<fvm_index_type> cv_children;      // CV child indices, partitioned by CV, and then in order.
    std::vector<fvm_index_type> cv_children_divs; // Paritions cv_children by CV index.

    std::vector<util::pw_elements<fvm_size_type>> branch_cv_map;     // CV offset map by branch.

    friend cv_geometry;
    friend cell_cv_geometry cell_cv_geometry_from_ends(const cable_cell& cell, const locset& lset);
};

cell_cv_geometry cell_cv_geometry_from_ends(const cable_cell& cell, const locset& lset);
} //namespace arb