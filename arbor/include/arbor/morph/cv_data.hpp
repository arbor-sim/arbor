#pragma once

#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>

namespace arb {
class cv_geometry;

// Stores info about the CV geometry of a discretized cable-cell
class cell_cv_data {
public:
    // Returns mcables comprising the CV at a given index.
    mcable_list cables(fvm_size_type cv_index) const;

    // Returns the CV indices of the children of a given CV index.
    std::vector<fvm_index_type> children(fvm_size_type cv_index) const;

    // Returns the CV index of the parent of a given CV index.
    fvm_index_type parent(fvm_size_type cv_index) const;

    // Returns total number of CVs.
    fvm_size_type num_cv() const;

private:
    std::vector<mcable> cv_cables;                // CV unbranched sections, partitioned by CV.
    std::vector<fvm_index_type> cv_cables_divs;   // Partitions cv_cables by CV index.

    std::vector<fvm_index_type> cv_parent;        // Index of CV parent or size_type(-1) for a cell root CV.
    std::vector<fvm_index_type> cv_children;      // CV child indices, partitioned by CV, and then in order.
    std::vector<fvm_index_type> cv_children_divs; // Paritions cv_children by CV index.

    friend cv_geometry;
    friend cell_cv_data cv_data_from_locset(const cable_cell& cell, const locset& lset);
};

class region_cv_data {
public:
    // Returns mcables comprising the CV at a given index.
    mcable_list cables(fvm_size_type cv_index) const;

    // Returns proportion of CV in the region, by area.
    fvm_value_type proportion(fvm_size_type cv_index) const;

    // Returns total number of CVs.
    fvm_size_type num_cv() const;

private:
    std::vector<mcable> cv_cables;                // CV unbranched sections, partitioned by CV.
    std::vector<fvm_index_type> cv_cables_divs;   // Partitions cv_cables by CV index
    std::vector<fvm_value_type> cv_proportion;    // Proportion of CV by area.

    friend region_cv_data intersect_region(const cable_cell& cell, const region& reg, const cell_cv_data& cvs);
};

// Construct cell_cv_geometry for cell from default cell discretization if it exists.
std::optional<cell_cv_data> cv_data(const cable_cell& cell);

// Construct cell_cv_geometry for cell from locset describing CV boundary points.
cell_cv_data cv_data_from_locset(const cable_cell& cell, const locset& lset);

region_cv_data intersect_region(const cable_cell& cell, const region& reg, const cell_cv_data& cvs);

} //namespace arb