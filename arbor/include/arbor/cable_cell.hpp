#pragma once

#include <unordered_map>
#include <string>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/common_types.hpp>
#include <arbor/constants.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/segment.hpp>

namespace arb {

// Location specification for point processes.

struct segment_location {
    segment_location(cell_lid_type s, double l):
        segment(s), position(l)
    {
        arb_assert(position>=0. && position<=1.);
    }

     bool operator==(segment_location other) const {
        return segment==other.segment && position==other.position;
    }

    cell_lid_type segment;
    double position;
};

// Current clamp description for stimulus specification.

struct i_clamp {
    using value_type = double;

    value_type delay = 0;      // [ms]
    value_type duration = 0;   // [ms]
    value_type amplitude = 0;  // [nA]

    i_clamp(value_type delay, value_type duration, value_type amplitude):
        delay(delay), duration(duration), amplitude(amplitude)
    {}
};

// Probe type for cell descriptions.

struct cell_probe_address {
    enum probe_kind {
        membrane_voltage, membrane_current
    };

    segment_location location;
    probe_kind kind;
};

/// high-level abstract representation of a cell and its segments
class cable_cell {
public:
    using index_type = cell_lid_type;
    using size_type = cell_local_size_type;
    using value_type = double;
    using point_type = point<value_type>;

    using gap_junction_instance = segment_location;

    struct synapse_instance {
        segment_location location;
        mechanism_desc mechanism;
    };

    struct stimulus_instance {
        segment_location location;
        i_clamp clamp;
    };

    struct detector_instance {
        segment_location location;
        double threshold;
    };

    cable_cell_parameter_set default_parameters;

    /// Default constructor
    cable_cell();

    /// Copy constructor
    cable_cell(const cable_cell& other):
        default_parameters(other.default_parameters),
        parents_(other.parents_),
        stimuli_(other.stimuli_),
        synapses_(other.synapses_),
        gap_junction_sites_(other.gap_junction_sites_),
        spike_detectors_(other.spike_detectors_)
    {
        // unique_ptr's cannot be copy constructed, do a manual assignment
        segments_.reserve(other.segments_.size());
        for (const auto& s: other.segments_) {
            segments_.push_back(s->clone());
        }
    }

    /// Move constructor
    cable_cell(cable_cell&& other) = default;

    /// Return the kind of cell, used for grouping into cell_groups
    cell_kind get_cell_kind() const  {
        return cell_kind::cable;
    }

    /// add a soma to the cell
    /// radius must be specified
    soma_segment* add_soma(value_type radius, point_type center=point_type());

    /// add a cable
    /// parent is the index of the parent segment for the cable section
    /// cable is the segment that will be moved into the cell
    cable_segment* add_cable(index_type parent, segment_ptr&& cable);

    /// add a cable by constructing it in place
    /// parent is the index of the parent segment for the cable section
    /// args are the arguments to be used to consruct the new cable
    template <typename... Args>
    cable_segment* add_cable(index_type parent, Args&&... args);

    /// the number of segments in the cell
    size_type num_segments() const;

    bool has_soma() const;

    class segment* segment(index_type index);
    const class segment* parent(index_type index) const;
    const class segment* segment(index_type index) const;

    /// access pointer to the soma
    /// returns nullptr if the cell has no soma
    soma_segment* soma();
    const soma_segment* soma() const;

    /// access pointer to a cable segment
    /// will throw an cable_cell_error exception if
    /// the cable index is not valid
    cable_segment* cable(index_type index);
    const cable_segment* cable(index_type index) const;

    /// the total number of compartments over all segments
    size_type num_compartments() const;

    std::vector<segment_ptr> const& segments() const {
        return segments_;
    }

    /// return a vector with the compartment count for each segment in the cell
    std::vector<size_type> compartment_counts() const;

    //////////////////
    // stimuli
    //////////////////
    void add_stimulus(segment_location loc, i_clamp stim);

    std::vector<stimulus_instance>&
    stimuli() {
        return stimuli_;
    }

    const std::vector<stimulus_instance>&
    stimuli() const {
        return stimuli_;
    }

    //////////////////
    // synapses
    //////////////////
    void add_synapse(segment_location loc, mechanism_desc p)
    {
        synapses_.push_back(synapse_instance{loc, std::move(p)});
    }
    const std::vector<synapse_instance>& synapses() const {
        return synapses_;
    }

    //////////////////
    // gap-junction
    //////////////////
    void add_gap_junction(segment_location location)
    {
        gap_junction_sites_.push_back(location);
    }
    const std::vector<gap_junction_instance>& gap_junction_sites() const {
        return gap_junction_sites_;
    }

    //////////////////
    // spike detectors
    //////////////////
    void add_detector(segment_location loc, double threshold);

    std::vector<detector_instance>&
    detectors() {
        return spike_detectors_;
    }

    const std::vector<detector_instance>&
    detectors() const {
        return spike_detectors_;
    }

    // Checks that two cells have the same
    //  - number and type of segments
    //  - volume and area properties of each segment
    //  - number of compartments in each segment
    // (note: just used for testing: move to test code?)
    friend bool cell_basic_equality(const cable_cell&, const cable_cell&);

    // Public view of parent indices vector.
    const std::vector<index_type>& parents() const {
        return parents_;
    }

    // Approximate per-segment mean attenuation b(f) at given frequency f,
    // ignoring membrane resistance [1/µm].
    value_type segment_mean_attenuation(value_type frequency, index_type segidx,
        const cable_cell_parameter_set& global_defaults) const;

    // Estimate of length constant λ(f) = 1/2 · 1/b(f), following
    // Hines and Carnevale (2001), "NEURON: A Tool for Neuroscientists",
    // Neuroscientist 7, pp. 123-135.
    value_type segment_length_constant(value_type frequency, index_type segidx,
        const cable_cell_parameter_set& global_defaults) const
    {
        return 0.5/segment_mean_attenuation(frequency, segidx, global_defaults);
    }

private:
    void assert_valid_segment(index_type) const;

    // storage for connections
    std::vector<index_type> parents_;

    // the segments
    std::vector<segment_ptr> segments_;

    // the stimuli
    std::vector<stimulus_instance> stimuli_;

    // the synapses
    std::vector<synapse_instance> synapses_;

    // the gap_junctions
    std::vector<gap_junction_instance> gap_junction_sites_;

    // the sensors
    std::vector<detector_instance> spike_detectors_;
};

// create a cable by forwarding cable construction parameters provided by the user
template <typename... Args>
cable_segment* cable_cell::add_cable(cable_cell::index_type parent, Args&&... args)
{
    // check for a valid parent id
    if (parent>=num_segments()) {
        throw cable_cell_error("parent index of cell segment is out of range");
    }
    segments_.push_back(make_segment<cable_segment>(std::forward<Args>(args)...));
    parents_.push_back(parent);

    return segments_.back()->as_cable();
}

// Create a cable cell from a morphology specification.
// If compartments_from_discretization is true, set number of compartments
// in each segment to be the number of piecewise linear sections in the
// corresponding section of the morphology.
cable_cell make_cable_cell(const morphology& morph, bool compartments_from_discretization);

} // namespace arb
