#pragma once


#include <unordered_map>
#include <string>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/common_types.hpp>
#include <arbor/constants.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/segment.hpp>

namespace arb {

// Pair of indexes that describe range of local indices.
// Returned by cable_cell::place() calls, so that the caller can
// refer to targets, detectors, etc on the cell.
struct locrange {
    cell_lid_type first;
    cell_lid_type last;
    locrange(cell_lid_type first, cell_lid_type last):
        first(first), last(last) {}
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

    using region_map = std::unordered_map<std::string, mcable_list>;
    using locset_map = std::unordered_map<std::string, mlocation_list>;

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

    /// add a soma to the cell
    /// radius must be specified
    soma_segment* add_soma(value_type radius, point_type center=point_type());

    /// add a cable
    /// parent is the index of the parent segment for the cable section
    /// cable is the segment that will be moved into the cell
    cable_segment* add_cable(index_type parent, segment_ptr&& cable);

    bool has_soma() const;

    class segment* segment(index_type index);
    const class segment* parent(index_type index) const;
    const class segment* segment(index_type index) const;

    // access pointer to the soma
    // returns nullptr if the cell has no soma
    // LEGACY
    soma_segment* soma();
    const soma_segment* soma() const;

    // access pointer to a cable segment
    // will throw an cable_cell_error exception if
    // the cable index is not valid
    // LEGACY
    cable_segment* cable(index_type index);
    const cable_segment* cable(index_type index) const;

    const std::vector<segment_ptr>& segments() const;

    // the number of segments in the cell
    size_type num_segments() const;

    // return a vector with the compartment count for each segment in the cell
    // LEGACY
    std::vector<size_type> compartment_counts() const;

    // The total number of compartments in the discretised cell.
    // LEGACY
    size_type num_compartments() const;

    //
    // Painters and placers.
    //
    // Used to describe regions and locations where density channels, stimuli,
    // synapses, gap juncitons and detectors are located.
    //

    // Density channels.
    void paint(const std::string& target, mechanism_desc);

    // Synapses.
    locrange place(const std::string& target, mechanism_desc);
    locrange place(mlocation, mechanism_desc);

    // Stimuli.
    locrange place(const std::string& target, i_clamp stim);
    locrange place(mlocation, i_clamp stim);

    // Gap junctions.
    locrange place(const std::string&, gap_junction_site);
    locrange place(mlocation, gap_junction_site);

    // spike detectors
    locrange place(const std::string&, detector);
    locrange place(mlocation, detector);

    //
    // access to placed items
    //

    const std::vector<synapse_instance>& synapses() const;
    const std::vector<gap_junction_instance>& gap_junction_sites() const;
    const std::vector<detector_instance>& detectors() const;
    const std::vector<stimulus_instance>& stimuli() const;

    // These setters are temporary, for "side-loading" in make_cable_cell.
    // In the regions, locset and morphology descriptions will be passed directly
    // to the cable_cell constructor.
    void set_regions(region_map r);
    void set_locsets(locset_map l);
    void set_morphology(em_morphology m);

    const em_morphology* morphology() const;

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

// Create a cable cell from a morphology specification.
// If compartments_from_discretization is true, set number of compartments
// in each segment to be the number of piecewise linear sections in the
// corresponding section of the morphology.
cable_cell make_cable_cell(const morphology& morph,
                           const label_dict& labels={},
                           bool compartments_from_discretization=false);

} // namespace arb
