#pragma once

#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/mechinfo.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/recipe.hpp>
#include <arbor/morph/cv_data.hpp>

#include "execution_context.hpp"
#include "morph/cv_data.hpp"
#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {

// CV geometry as determined by per-cell CV boundary points.
//
// Details of CV cable representation:
//
//   * The extent of the cables of a control volume corresponds to the
//     closure of the control volume on the morphology tree.
//
//   * Every fork in the morphology tree 'belongs' to exactly
//     one CV. A fork belongs to a CV if and only if there is
//     a cable in the CV for each branch of the fork, that
//     includes the fork point.
//
//   * If a CV has more than one cable covering a fork point, then
//     that fork point must belong to that CV.
//
// These requirements are a consequence of the CV geometry being determined
// by a collection of CV boundarary locations.

namespace cv_prefer {
    // Enum for resolving which CV to return on location look-up, if the
    // location is on a CV boundary.

    enum type {
        // Prefer more proximal CV:
        cv_proximal,
        // Prefer more distal CV:
        cv_distal,
        // Prefer distal CV unless it has zero extent on location branch.
        // This should be used for placing point processes on CVs.
        cv_nonempty,
        // Prefer distal CV unless the proximal CV has zero extent on location branch.
        // This should be used for determing to which CV a fork point belongs.
        cv_empty
    };
}

struct ARB_ARBOR_API cv_geometry: public cell_cv_data_impl {
    using base = cell_cv_data_impl;

    using size_type = arb_size_type;
    using index_type = arb_index_type;

    std::vector<index_type> cv_to_cell;      // Maps CV index to cell index.
    std::vector<index_type> cell_cv_divs;    // Partitions CV indices by cell.

    // CV offset map by cell index then branch. Used for location_cv query.
    std::vector<std::vector<util::pw_elements<size_type>>> branch_cv_map;

    cv_geometry() = default;

    auto cables(size_type cv_index) const {
        auto partn = util::partition_view(cv_cables_divs);
        return util::subrange_view(cv_cables, partn[cv_index]);
    }

    auto children(size_type cv_index) const {
        auto partn = util::partition_view(cv_children_divs);
        return util::subrange_view(cv_children, partn[cv_index]);
    }

    std::pair<index_type, index_type> cell_cv_interval(size_type cell_idx) const {
        auto partn = util::partition_view(cell_cv_divs);
        return partn[cell_idx];
    }

    auto cell_cvs(size_type cell_idx) const {
        auto partn = util::partition_view(cell_cv_divs);
        return util::make_span(partn[cell_idx]);
    }

    size_type size() const {
        arb_assert((cv_parent.empty() && cv_cables_divs.empty() &&
                    cv_cables.empty() && cv_to_cell.empty())
                   ||
                   (cv_parent.size()+1 == cv_cables_divs.size() &&
                    cv_parent.size() == cv_to_cell.size() &&
                    (unsigned)cv_to_cell.back()+1 == cell_cv_divs.size()-1));

        return cv_parent.size();
    }

    bool empty() const {
        return size()==0;
    }

    size_type n_cell() const {
        return cell_cv_divs.empty()? 0: cell_cv_divs.size()-1;
    }

    size_type n_branch(size_type cell_idx) const {
        return branch_cv_map.at(cell_idx).size();
    }

    size_type location_cv(size_type cell_idx, const mlocation& loc, cv_prefer::type prefer) const;

    cv_geometry(const cable_cell& cell, const locset& ls);
};

// Combine two cv_geometry groups in-place.
// (Returns reference to first argument.)
ARB_ARBOR_API cv_geometry& append(cv_geometry&, const cv_geometry&);

// Discretization of morphologies and physical properties. Contains cv_geometry
// as above.
//
// diam_um is taken to be the diameter of a CV with constant diameter and same
// extent which has the same surface area (i.e. cv_area/(πL) where L is the
// total length of the cables comprising the CV.)
//
// The bulk conductivity over the morphology is recorded here as well, for
// calculating voltage and axial current interpolating probes.
//
// For the computation of inter-CV conductances and voltage interpolation, it
// is assumed that the CV voltage is exact at every fork point that belongs
// to the CV, or in the absence of any internal forks, is exact at the
// midpoint of an unbranched CV.

struct fvm_diffusion_info {
    using value_type = arb_value_type;
    std::vector<value_type> face_diffusivity;
    std::vector<std::vector<pw_constant_fn>> axial_inv_diffusivity;
};
    
struct fvm_cv_discretization {
    using size_type = arb_size_type;
    using index_type = arb_index_type;
    using value_type = arb_value_type;

    cv_geometry geometry;

    bool empty() const { return geometry.empty(); }
    size_type size() const { return geometry.size(); }
    size_type n_cell() const { return geometry.n_cell(); }

    // Following members have one element per CV.
    std::vector<value_type> face_conductance; // [µS]
    std::vector<value_type> cv_area;          // [µm²]
    std::vector<value_type> cv_capacitance;   // [pF]
    std::vector<value_type> init_membrane_potential; // [mV]
    std::vector<value_type> temperature_K;    // [K]
    std::vector<value_type> diam_um;          // [µm]

    // For each cell, one piece-wise constant value per branch.
    std::vector<std::vector<pw_constant_fn>> axial_resistivity; // [Ω·cm]

    // For each diffusive ion species, their properties
    std::unordered_map<std::string, fvm_diffusion_info> diffusive_ions;
};

// Combine two fvm_cv_geometry groups in-place.
// (Returns reference to first argument.)
ARB_ARBOR_API fvm_cv_discretization& append(fvm_cv_discretization&, const fvm_cv_discretization&);

// Construct fvm_cv_discretization from one or more cells.
ARB_ARBOR_API fvm_cv_discretization fvm_cv_discretize(const cable_cell& cell, const cable_cell_parameter_set& global_dflt);
ARB_ARBOR_API fvm_cv_discretization fvm_cv_discretize(const std::vector<cable_cell>& cells, const cable_cell_parameter_set& global_defaults, const arb::execution_context& ctx={});


// Interpolant data for voltage, axial current probes.
//
// The proximal CV will always be either the same as distal CV (trivial
// interpolation) or the parent of the distal CV.

struct fvm_voltage_interpolant {
    arb_index_type proximal_cv, distal_cv;
    arb_value_type proximal_coef, distal_coef;
};

// Interpolated membrane voltage.
ARB_ARBOR_API fvm_voltage_interpolant fvm_interpolate_voltage(const cable_cell& cell, const fvm_cv_discretization& D, arb_size_type cell_idx, const mlocation& site);

// Axial current as linear combiantion of voltages.
ARB_ARBOR_API fvm_voltage_interpolant fvm_axial_current(const cable_cell& cell, const fvm_cv_discretization& D, arb_size_type cell_idx, const mlocation& site);


// Post-discretization data for point and density mechanism instantiation.

struct fvm_mechanism_config {
    using value_type = arb_value_type;
    using index_type = arb_index_type;

    arb_mechanism_kind kind;

    // Ordered CV indices where mechanism is present; may contain
    // duplicates for point mechanisms.
    std::vector<index_type> cv;

    // Coalesced synapse multiplier (point mechanisms only).
    std::vector<index_type> multiplicity;

    // Normalized area contribution in corresponding CV (density mechanisms only).
    std::vector<value_type> norm_area;

    // Synapse target number (point mechanisms only).
    // For each instance index i, there are multiplicity[i] consecutive entries.
    std::vector<index_type> target;

    // Gap junction peer CV index (gap junction mechanisms only)
    std::vector<index_type> peer_cv;

    // Gap junction weight, unit-less (gap junction mechanisms only)
    std::vector<value_type> local_weight;

    // (Non-global) parameters and parameter values across the mechanism instance.
    std::vector<std::pair<std::string, std::vector<value_type>>> param_values;
};

// Post-discretization data for ion channel state.

struct fvm_ion_config {
    using value_type = arb_value_type;
    using index_type = arb_index_type;

    // Keep track whether eX, Xi, Xo are actually to be reset.
    bool revpot_written = false;
    bool iconc_written = false;
    bool econc_written = false;

    // Ordered CV indices where ion must be present.
    std::vector<index_type> cv;

    // Normalized area contribution of default concentration contribution in corresponding CV.
    std::vector<value_type> init_iconc;
    std::vector<value_type> init_econc;

    // Normalized area contribution of default concentration contribution in corresponding CV set by users
    std::vector<value_type> reset_iconc;
    std::vector<value_type> reset_econc;

    // Ion-specific (initial) reversal potential per CV.
    std::vector<value_type> init_revpot;

    // diffusivity
    bool is_diffusive = false;
    std::vector<value_type> face_diffusivity;
};

struct fvm_stimulus_config {
    using value_type = arb_value_type;
    using index_type = arb_index_type;

    // CV index for each stimulus instance; monotonically increasing.
    std::vector<index_type> cv;

    // CV indices where one or more stimuli are present; strictly monotonically increasing.
    std::vector<index_type> cv_unique;

    // Frequency, amplitude info, per instance.
    // Note that amplitudes have been scaled by 1/CV area so that they are represent as current densities, not currents.
    std::vector<double> frequency; // [kHz]
    std::vector<double> phase; // [rad]
    std::vector<std::vector<double>> envelope_time;      // [ms]
    std::vector<std::vector<double>> envelope_amplitude; // [A/m²]
};

// Maps gj {gid, lid} locations on a cell to their CV indices.
ARB_ARBOR_API std::unordered_map<cell_member_type, arb_size_type> fvm_build_gap_junction_cv_map(
    const std::vector<cable_cell>& cells,
    const std::vector<cell_gid_type>& gids,
    const fvm_cv_discretization& D);

// Resolves gj_connections into {gid, lid} pairs, then to CV indices and a weight.
ARB_ARBOR_API std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> fvm_resolve_gj_connections(
    const std::vector<cell_gid_type>& gids,
    const cell_label_range& gj_data,
    const std::unordered_map<cell_member_type, arb_size_type>& gj_cv,
    const recipe& rec);

struct fvm_mechanism_data {
    // Mechanism config, indexed by mechanism name.
    std::map<std::string, fvm_mechanism_config> mechanisms;

    // Ion config, indexed by ion name.
    std::unordered_map<std::string, fvm_ion_config> ions;

    // Stimulus config.
    fvm_stimulus_config stimuli;

    // Total number of targets (point-mechanism points).
    std::size_t n_target = 0;

    // Partitions target numbers by cell.
    std::vector<std::size_t> target_divs;

    // Contains mechanisms with post_event
    bool post_events = false;
};

ARB_ARBOR_API fvm_mechanism_data fvm_build_mechanism_data(
    const cable_cell_global_properties& gprop,
    const std::vector<cable_cell>& cells,
    const std::vector<cell_gid_type>& gids,
    const std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>>& gj_conns,
    const fvm_cv_discretization& D,
    const arb::execution_context& ctx={});

} // namespace arb
