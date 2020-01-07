#pragma once

#include <string>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/common_types.hpp>
#include <arbor/constants.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/segment.hpp>

namespace arb {

// Pair of indexes that describe range of local indices.
// Returned by cable_cell::place() calls, so that the caller can
// refer to targets, detectors, etc on the cell.
struct lid_range {
    cell_lid_type begin;
    cell_lid_type end;
    lid_range(cell_lid_type b, cell_lid_type e):
        begin(b), end(e) {}
};

// Probe type for cell descriptions.
struct cell_probe_address {
    enum probe_kind {
        membrane_voltage, membrane_current
    };

    mlocation location;
    probe_kind kind;
};

// Forward declare the implementation, for PIMPL.
struct cable_cell_impl;

// High-level abstract representation of a cell and its segments
class cable_cell {
public:
    using index_type = cell_lid_type;
    using size_type = cell_local_size_type;
    using value_type = double;
    using point_type = point<value_type>;

    using gap_junction_instance = mlocation;

    struct synapse_instance {
        mlocation location;
        mechanism_desc mechanism;
    };

    struct stimulus_instance {
        mlocation location;
        i_clamp clamp;
    };

    struct detector_instance {
        mlocation location;
        double threshold;
    };

    cable_cell_parameter_set default_parameters;

    /// Default constructor
    cable_cell();

    /// Copy constructor
    cable_cell(const cable_cell& other);

    /// Move constructor
    cable_cell(cable_cell&& other) = default;

    /// construct from morphology
    cable_cell(const class morphology& m,
               const label_dict& dictionary={},
               bool compartments_from_discretization=false);

    /// Access to morphology and embedding
    const concrete_embedding& embedding() const;
    const arb::morphology& morphology() const;
    const mprovider& provider() const;

    // the number of branches in the cell
    size_type num_branches() const;

    // All of the members marked with LEGACY below will be removed once
    // the discretization code has moved from consuming segments to em_morphology.

    // LEGACY
    bool has_soma() const;

    // LEGACY
    const class segment* parent(index_type index) const;
    // LEGACY
    const class segment* segment(index_type index) const;

    // access pointer to the soma
    // returns nullptr if the cell has no soma
    // LEGACY
    const soma_segment* soma() const;

    // access pointer to a cable segment
    // will throw an cable_cell_error exception if
    // the cable index is not valid
    // LEGACY
    const cable_segment* cable(index_type index) const;

    // LEGACY
    const std::vector<segment_ptr>& segments() const;

    // return a vector with the compartment count for each segment in the cell
    // LEGACY
    std::vector<size_type> compartment_counts() const;

    // The total number of compartments in the discretised cell.
    // LEGACY
    size_type num_compartments() const;

    // Painters and placers.
    //
    // Used to describe regions and locations where density channels, stimuli,
    // synapses, gap juncitons and detectors are located.

    // Density channels.
    void paint(const region&, mechanism_desc);

    // Properties.
    void paint(const region&, cable_cell_local_parameter_set);

    // Synapses.
    lid_range place(const locset&, const mechanism_desc&);

    // Stimuli.
    lid_range place(const locset&, const i_clamp&);

    // Gap junctions.
    lid_range place(const locset&, gap_junction_site);

    // Spike detectors.
    lid_range place(const locset&, const threshold_detector&);

    // Access to placed items.

    const std::vector<synapse_instance>& synapses() const;
    const std::vector<gap_junction_instance>& gap_junction_sites() const;
    const std::vector<detector_instance>& detectors() const;
    const std::vector<stimulus_instance>& stimuli() const;

    // Checks that two cells have the same
    //  - number and type of segments
    //  - volume and area properties of each segment
    //  - number of compartments in each segment
    // (note: just used for testing: move to test code?)
    friend bool cell_basic_equality(const cable_cell&, const cable_cell&);

    // Public view of parent indices vector.
    const std::vector<index_type>& parents() const;

    // Approximate per-segment mean attenuation b(f) at given frequency f,
    // ignoring membrane resistance [1/µm].
    value_type segment_mean_attenuation(value_type frequency, index_type segidx,
        const cable_cell_parameter_set& global_defaults) const;

    // Estimate of length constant λ(f) = 1/2 · 1/b(f), following
    // Hines and Carnevale (2001), "NEURON: A Tool for Neuroscientists",
    // Neuroscientist 7, pp. 123-135.
    value_type segment_length_constant(value_type frequency, index_type segidx,
        const cable_cell_parameter_set& global_defaults) const;

private:
    std::unique_ptr<cable_cell_impl, void (*)(cable_cell_impl*)> impl_;
};

} // namespace arb
