#pragma once

#include <vector>

#include <arbor/export.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>

namespace arb {

struct cell_cv_data_impl;

// Stores info about the CV geometry of a discretized cable-cell
class ARB_ARBOR_API cell_cv_data {
public:
    // Returns mcables comprising the CV at a given index.
    mcable_list cables(arb_size_type index) const;

    // Returns the CV indices of the children of a given CV index.
    std::vector<arb_index_type> children(arb_size_type index) const;

    // Returns the CV index of the parent of a given CV index.
    arb_index_type parent(arb_size_type index) const;

    // Returns total number of CVs.
    arb_size_type size() const;

    cell_cv_data(const cable_cell& cell, const locset& lset);

    const mprovider& provider() const {
        return provider_;
    }

private:
    std::unique_ptr<cell_cv_data_impl, void (*)(cell_cv_data_impl*)> impl_;

    // Embedded morphology and labelled region/locset lookup.
    mprovider provider_;
};

struct cv_proportion {
    arb_size_type idx;
    arb_value_type proportion;
};

// Construct cell_cv_geometry for cell from default cell discretization if it exists.
ARB_ARBOR_API std::optional<cell_cv_data> cv_data(const cable_cell& cell);

ARB_ARBOR_API std::vector<cv_proportion> intersect_region(const region& reg, const cell_cv_data& cvs, bool intergrate_by_length = false);

} //namespace arb
