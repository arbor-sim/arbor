#pragma once

#include <backends/fvm_types.hpp>
#include <cell.hpp>
#include <compartment.hpp>
#include <mechanism.hpp>
#include <mechinfo.hpp>
#include <mechcat.hpp>
#include <util/deduce_return.hpp>
#include <util/enumhash.hpp>
#include <util/span.hpp>

namespace arb {

// Discretization data for an unbranched segment.

struct segment_info {
    using value_type = fvm_value_type;
    using size_type = fvm_size_type;

    value_type parent_cv_area = 0;
    value_type distal_cv_area = 0;

    static constexpr size_type npos = (size_type)-1;

    size_type parent_cv = npos; // npos => no parent.
    size_type proximal_cv = 0;  // First CV in segment, excluding parent.
    size_type distal_cv = 0;    // Last CV in segment (may be shared with other segments).

    bool has_parent() const { return parent_cv!=npos; }

    // Range of CV-indices for segment, excluding parent.
    std::pair<size_type, size_type> cv_range() const {
        return {proximal_cv, 1+distal_cv};
    }

    // Position is proportional distal distance along segment, in [0, 1).
    size_type cv_by_position(double pos) const {
        size_type n = distal_cv+1-proximal_cv;
        size_type i = static_cast<size_type>(n*pos+0.5);
        if (i>0) {
            return proximal_cv+(i-1);
        }
        else {
            return parent_cv==npos? proximal_cv: parent_cv;
        }
    }
};

// Discretization of morphologies and electrical properties for
// cells in a cell group.

struct fvm_discretization {
    using value_type = fvm_value_type;
    using size_type = fvm_size_type;

    size_type ncell;
    size_type ncomp;

    // Note: if CV j has no parent, parent_cv[j] = j. TODO: confirm!
    std::vector<size_type> parent_cv;
    std::vector<size_type> cv_to_cell;

    std::vector<value_type> face_conductance; // [µS]
    std::vector<value_type> cv_area;          // [µm²]
    std::vector<value_type> cv_capacitance;   // [pF]

    std::vector<segment_info> segments;
    std::vector<size_type> cell_segment_bounds; // Partitions segment indices by cell.
    std::vector<size_type> cell_cv_bounds;      // Partitions CV indices by cell.

    auto cell_segment_part() const
        DEDUCED_RETURN_TYPE(util::partition_view(cell_segment_bounds))

    auto cell_cv_part() const
        DEDUCED_RETURN_TYPE(util::partition_view(cell_cv_bounds))

    size_type segment_location_cv(size_type cell_index, segment_location segloc) const {
        auto cell_segs = cell_segment_part()[cell_index];

        size_type seg = segloc.segment+cell_segs.first;
        EXPECTS(seg<cell_segs.second);
        return segments[seg].cv_by_position(segloc.position);
    }
};

fvm_discretization fvm_discretize(const std::vector<cell>& cells);


// Post-discretization data for point and density mechanism instantiation.

struct fvm_mechanism_config {
    using value_type = fvm_value_type;
    using size_type = fvm_size_type;

    mechanismKind kind;

    // Ordered CV indices where mechanism is present; may contain
    // duplicates for point mechanisms.
    std::vector<size_type> cv;

    // Normalized area contribution in corresponding CV (density mechanisms only).
    std::vector<value_type> norm_area;

    // Synapse target number (point mechanisms only).
    std::vector<size_type> target;

    // (Non-global) parameters and parameter values across the mechanism instance.
    std::vector<std::pair<std::string, std::vector<value_type>>> param_values;
};

// Post-discretization data for ion channel state.

struct fvm_ion_config {
    using value_type = fvm_value_type;
    using size_type = fvm_size_type;

    // Ordered CV indices where ion must be present.
    std::vector<size_type> cv;

    // Normalized area contribution of default concentration contribution in corresponding CV.
    std::vector<value_type> iconc_norm_area;
    std::vector<value_type> econc_norm_area;
};

struct fvm_mechanism_data {
    // Mechanism config, indexed by mechanism name.
    std::unordered_map<std::string, fvm_mechanism_config> mechanisms;

    // Ion config, indexed by ionKind.
    std::unordered_map<ionKind, fvm_ion_config, util::enum_hash> ions;

    // Total number of targets (point-mechanism points)
    std::size_t ntarget = 0;
};

fvm_mechanism_data fvm_build_mechanism_data(const mechanism_catalogue& catalogue, const std::vector<cell>& cells, const fvm_discretization& D);

} // namespace arb
