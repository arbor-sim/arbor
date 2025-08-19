#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <type_traits>

#include <arbor/export.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/common_types.hpp>
#include <arbor/constants.hpp>
#include <arbor/iexpr.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/mcable_map.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/util/hash_def.hpp>
#include <arbor/util/extra_traits.hpp>

namespace arb {

// Metadata for point process probes.
struct ARB_SYMBOL_VISIBLE cable_probe_point_info {
    cell_tag_type target;   // Target tag of point process instance on cell.
    cell_lid_type lid;      // Target lid of point process instance on cell.
    unsigned multiplicity;  // Number of combined instances at this site.
    mlocation loc;          // Point on cell morphology where instance is placed.
};

// Cable cell type definitions
using cable_sample_type = const double;

using cable_state_meta_type = const mlocation;
using cable_state_cell_meta_type = const mcable;
using cable_point_meta_type = const cable_probe_point_info;

template <>
struct probe_value_type_of<cable_state_meta_type> {
    using type = cable_sample_type;
};

template <>
struct probe_value_type_of<cable_state_cell_meta_type> {
    using type = cable_sample_type;
};

template <>
struct probe_value_type_of<cable_point_meta_type> {
    using type = cable_sample_type;
};

// Each kind of probe has its own type for representing its address, as below.
// The metadata associated with a probe is also passed to a sampler via an `any_ptr`;
// the underlying pointer will be a const pointer to the associated metadata type.

// Voltage estimate [mV] at `location`,
// interpolated.
struct ARB_SYMBOL_VISIBLE cable_probe_membrane_voltage {
    using value_type = cable_sample_type;
    using meta_type = cable_state_meta_type;
    locset locations;
};

// Voltage estimate [mV], reported against each cable in each control volume.
// Not interpolated.
struct ARB_SYMBOL_VISIBLE cable_probe_membrane_voltage_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
};

// Axial current estimate [nA] at `location`,
// interpolated.
struct ARB_SYMBOL_VISIBLE cable_probe_axial_current {
    using value_type = cable_sample_type;
    using meta_type = cable_state_meta_type;
    locset locations;
};

// Total current density [A/m²] across membrane _excluding_ capacitive and
// stimulus current at `location`.
struct ARB_SYMBOL_VISIBLE cable_probe_total_ion_current_density {
    using value_type = cable_sample_type;
    using meta_type = cable_state_meta_type;
    locset locations;
};

// Total ionic current [nA] across membrane _excluding_ capacitive current across components of the cell.
// Sample value type: `cable_sample_range`
// Sample metadata type: `mcable_list`
struct ARB_SYMBOL_VISIBLE cable_probe_total_ion_current_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
};

// Total membrane current [nA] across components of the cell _excluding_ stimulus currents.
struct ARB_SYMBOL_VISIBLE cable_probe_total_current_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
};

// Stimulus currents [nA] across components of the cell.
struct ARB_SYMBOL_VISIBLE cable_probe_stimulus_current_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
};

// Value of state variable `state` in density mechanism `mechanism` in CV at `location`.
struct ARB_SYMBOL_VISIBLE cable_probe_density_state {
    using value_type = cable_sample_type;
    using meta_type = cable_state_meta_type;
    locset locations;
    std::string mechanism;
    std::string state;
};

// Value of state variable `state` in density mechanism `mechanism` across components of the cell.
struct ARB_SYMBOL_VISIBLE cable_probe_density_state_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
    std::string mechanism;
    std::string state;
};

// Value of state variable `key` in point mechanism `source` at target `target`.
struct ARB_SYMBOL_VISIBLE cable_probe_point_state {
    using value_type = cable_sample_type;
    using meta_type = cable_point_meta_type;

    cell_tag_type target;
    std::string mechanism;
    std::string state;

    // Engage in minimal hygeine. Ideally, we'd disable all nullptr constructors.
    cable_probe_point_state(std::nullptr_t, std::string, std::string) = delete;
    cable_probe_point_state() = delete;

    constexpr cable_probe_point_state(cell_tag_type t, std::string m, std::string s):
        target(std::move(t)), mechanism(std::move(m)), state(std::move(s)) {}
    constexpr cable_probe_point_state(const cable_probe_point_state&) = default;
    constexpr cable_probe_point_state(cable_probe_point_state&&) = default;
    constexpr cable_probe_point_state& operator=(const cable_probe_point_state&) = default;
    constexpr cable_probe_point_state& operator=(cable_probe_point_state&&) = default;
};

// Value of state variable `key` in point mechanism `source` at every target
// with this mechanism. Metadata has one entry of type cable_probe_point_info
// for each matched (possibly coalesced) instance.
struct ARB_SYMBOL_VISIBLE cable_probe_point_state_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_point_meta_type;
    std::string mechanism;
    std::string state;
};

// Current density [A/m²] across membrane attributed to the ion `source` at `location`.
struct ARB_SYMBOL_VISIBLE cable_probe_ion_current_density {
    using value_type = cable_sample_type;
    using meta_type = cable_state_meta_type;
    locset locations;
    std::string ion;
};

// Total ionic current [nA] attributed to the ion `source` across components of the cell.
struct ARB_SYMBOL_VISIBLE cable_probe_ion_current_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
    std::string ion;
};

// Ionic internal concentration [mmol/L] of ion `source` at `location`.
struct ARB_SYMBOL_VISIBLE cable_probe_ion_int_concentration {
    using value_type = cable_sample_type;
    using meta_type = cable_state_meta_type;
    locset locations;
    std::string ion;
};

// Ionic internal concentration [mmol/L] of ion `source` across components of the cell.
struct ARB_SYMBOL_VISIBLE cable_probe_ion_int_concentration_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
    std::string ion;
};

// Ionic diffusive concentration [mmol/L] of ion `source` at `location`.
struct ARB_SYMBOL_VISIBLE cable_probe_ion_diff_concentration {
    using value_type = cable_sample_type;
    using meta_type = cable_state_meta_type;
    locset locations;
    std::string ion;
};

// Ionic diffusive concentration [mmol/L] of ion `source` across components of the cell.
struct ARB_SYMBOL_VISIBLE cable_probe_ion_diff_concentration_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
    std::string ion;
};

// Ionic external concentration [mmol/L] of ion `source` at `location`.
struct ARB_SYMBOL_VISIBLE cable_probe_ion_ext_concentration {
    using value_type = cable_sample_type;
    using meta_type = cable_state_meta_type;
    locset locations;
    std::string ion;
};

// Ionic external concentration [mmol/L] of ion `source` across components of the cell.
struct ARB_SYMBOL_VISIBLE cable_probe_ion_ext_concentration_cell {
    using value_type = cable_sample_type;
    using meta_type = cable_state_cell_meta_type;
    std::string ion;
};

// Forward declare the implementation, for PIMPL.
struct cable_cell_impl;

// Typed maps for access to painted and placed assignments:
//
// Mechanisms and initial ion data are further keyed by mechanism name and ion
// name respectively.
using iexpr_map = std::unordered_map<std::string, iexpr_ptr>;

template <typename T>
using region_assignment = std::conditional_t<std::is_same_v<T, init_int_concentration> || std::is_same_v<T, init_ext_concentration> || std::is_same_v<T, init_reversal_potential> || std::is_same_v<T, ion_diffusivity> || std::is_same_v<T, voltage_process>,
                                             std::unordered_map<std::string, mcable_map<T>>,
                                             std::conditional_t<std::is_same_v<T, density>,
                                                                std::unordered_map<std::string, mcable_map<std::pair<T, iexpr_map>>>,
                                                                mcable_map<T>>>;

template <typename T>
struct placed {
    mlocation loc;
    cell_lid_type lid;
    T item;
    hash_type tag;
};

// Note: lid fields of elements of mlocation_map used in cable_cell are strictly increasing.
template <typename T>
using mlocation_map = std::vector<placed<T>>;

// map for synapses or junctions, plain vector else
template <typename T>
using location_assignment = std::conditional_t<std::is_same_v<T, synapse> || std::is_same_v<T, junction>,
                                               std::unordered_map<std::string, mlocation_map<T>>,
                                               mlocation_map<T>>;

// High-level abstract representation of a cell.
struct ARB_SYMBOL_VISIBLE cable_cell {
    using lid_range_map = std::unordered_multimap<hash_type, lid_range>;
    using index_type = cell_lid_type;
    using size_type = cell_local_size_type;
    using value_type = double;

    // Default constructor.
    cable_cell();

    // Copy and move constructors.
    cable_cell(const cable_cell& other);
    cable_cell(cable_cell&& other) = default;

    // Copy and move assignment operators.
    cable_cell& operator=(cable_cell&&) = default;
    cable_cell& operator=(const cable_cell& other) { return *this = cable_cell(other); }

    /// Construct from morphology, label and decoration descriptions.
    cable_cell(const class morphology& m,
               const decor& d,
               const label_dict& l={},
               const std::optional<cv_policy>& = {});

    /// Access to labels
    const label_dict& labels() const;

    /// Access to morphology and embedding
    const concrete_embedding& embedding() const;
    const arb::morphology& morphology() const;
    const mprovider& provider() const;

    // Convenience access to placed items.
    const std::unordered_map<std::string, mlocation_map<synapse>>& synapses() const;
    const std::unordered_map<std::string, mlocation_map<junction>>& junctions() const;
    const mlocation_map<threshold_detector>& detectors() const;
    const mlocation_map<i_clamp>& stimuli() const;

    // Convenience access to painted items.
    const region_assignment<density> densities() const;
    const region_assignment<voltage_process> voltage_processes() const;
    const region_assignment<init_int_concentration> init_int_concentrations() const;
    const region_assignment<init_ext_concentration> init_ext_concentrations() const;
    const region_assignment<init_reversal_potential> reversal_potentials() const;
    const region_assignment<ion_diffusivity> diffusivities() const;
    const region_assignment<temperature> temperatures() const;
    const region_assignment<init_membrane_potential> init_membrane_potentials() const;
    const region_assignment<axial_resistivity> axial_resistivities() const;
    const region_assignment<membrane_capacitance> membrane_capacitances() const;

    // Access to a concrete list of locations for a locset.
    mlocation_list concrete_locset(const locset&) const;
    // Access to a concrete list of cable segments for a region.
    mextent concrete_region(const region&) const;

    // The decorations on the cell.
    const decor& decorations() const;

    // The current cv_policy of this cell
    const std::optional<cv_policy>& discretization() const;
    void discretization(cv_policy);

    // The default parameter and ion settings on the cell.
    const cable_cell_parameter_set& default_parameters() const;

    // The labeled lid_ranges of sources, targets and gap_junctions on the cell;
    const lid_range_map& detector_ranges() const;
    const lid_range_map& synapse_ranges() const;
    const lid_range_map& junction_ranges() const;

private:
    std::unique_ptr<cable_cell_impl, void (*)(cable_cell_impl*)> impl_;
};

} // namespace arb

ARB_DEFINE_HASH(arb::cable_probe_point_info, a.target, a.multiplicity, a.loc);
