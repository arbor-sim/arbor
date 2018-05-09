#pragma once

#include <unordered_map>
#include <stdexcept>
#include <vector>

#include <common_types.hpp>
#include <constants.hpp>
#include <ion.hpp>
#include <mechcat.hpp>
#include <morphology.hpp>
#include <segment.hpp>
#include <tree.hpp>
#include <util/rangeutil.hpp>

namespace arb {

// Location specification for point processes.

struct segment_location {
    segment_location(cell_lid_type s, double l):
        segment(s), position(l)
    {
        EXPECTS(position>=0. && position<=1.);
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

// Global parameter type for cell descriptions.

struct cell_global_properties {
    const mechanism_catalogue* catalogue = &global_default_catalogue();

    // TODO: consider making some/all of the following parameters
    // cell or even segment-local.
    // 
    // Consider also a model-level dictionary of default values that
    // can be used to initialize per-cell-kind info?
    //
    // Defaults below chosen to match NEURON.

    // Ion species currently limited to just "ca", "na", "k".
    std::unordered_map<std::string, ion_info> ion_default = {
        {"ca", { ionKind::ca, 2, 5e-5, 2.  }},
        {"na", { ionKind::na, 1, 10.,  140.}},
        {"k",  { ionKind::k,  1, 54.4, 2.5 }}
    };

    double temperature_K = constant::hh_squid_temp; // [K]
    double init_membrane_potential_mV = -65; // [mV]
};

/// Wrapper around compartment layout information derived from a high level cell
/// description.

struct compartment_model {
    arb::tree tree;
    std::vector<tree::int_type> parent_index;
    std::vector<tree::int_type> segment_index;
};

/// high-level abstract representation of a cell and its segments
class cell {
public:
    using index_type = cell_lid_type;
    using size_type = cell_local_size_type;
    using value_type = double;
    using point_type = point<value_type>;

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

    /// Default constructor
    cell();

    /// Copy constructor
    cell(const cell& other):
        parents_(other.parents_),
        stimuli_(other.stimuli_),
        synapses_(other.synapses_),
        spike_detectors_(other.spike_detectors_)
    {
        // unique_ptr's cannot be copy constructed, do a manual assignment
        segments_.reserve(other.segments_.size());
        for (const auto& s: other.segments_) {
            segments_.push_back(s->clone());
        }
    }

    /// Move constructor
    cell(cell&& other) = default;

    /// Return the kind of cell, used for grouping into cell_groups
    cell_kind get_cell_kind() const  {
        return cell_kind::cable1d_neuron;
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
    const class segment* segment(index_type index) const;

    /// access pointer to the soma
    /// returns nullptr if the cell has no soma
    soma_segment* soma();
    const soma_segment* soma() const;

    /// access pointer to a cable segment
    /// will throw an std::out_of_range exception if
    /// the cable index is not valid
    cable_segment* cable(index_type index);

    /// the volume of the cell
    value_type volume() const;

    /// the surface area of the cell
    value_type area() const;

    /// the total number of compartments over all segments
    size_type num_compartments() const;

    std::vector<segment_ptr> const& segments() const {
        return segments_;
    }

    /// return a vector with the compartment count for each segment in the cell
    std::vector<size_type> compartment_counts() const;

    compartment_model model() const;

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
    friend bool cell_basic_equality(const cell&, const cell&);

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

    // the sensors
    std::vector<detector_instance> spike_detectors_;
};

// create a cable by forwarding cable construction parameters provided by the user
template <typename... Args>
cable_segment* cell::add_cable(cell::index_type parent, Args&&... args)
{
    // check for a valid parent id
    if(parent>=num_segments()) {
        throw std::out_of_range(
            "parent index of cell segment is out of range"
        );
    }
    segments_.push_back(make_segment<cable_segment>(std::forward<Args>(args)...));
    parents_.push_back(parent);

    return segments_.back()->as_cable();
}

// Create a cell from a morphology specification.
// If compartments_from_discretization is true, set number of compartments in
// each segment to be the number of piecewise linear sections in the corresponding
// section of the morphologu.
cell make_cell(const morphology&, bool compartments_from_discretization=false);

} // namespace arb
