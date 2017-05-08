#pragma once

#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <common_types.hpp>
#include <cell_interface.hpp>
#include <cell_tree.hpp>
#include <morphology.hpp>
#include <segment.hpp>
#include <stimulus.hpp>
#include <util/debug.hpp>
#include <util/pprintf.hpp>
#include <util/rangeutil.hpp>

namespace nest {
namespace mc {

/// wrapper around compartment layout information derived from a high level cell
/// description
struct compartment_model {
    cell_tree tree;
    std::vector<cell_tree::int_type> parent_index;
    std::vector<cell_tree::int_type> segment_index;
};

struct segment_location {
    segment_location(cell_lid_type s, double l)
    : segment(s), position(l)
    {
        EXPECTS(position>=0. && position<=1.);
    }
    friend bool operator==(segment_location l, segment_location r) {
        return l.segment==r.segment && l.position==r.position;
    }
    cell_lid_type segment;
    double position;
};

int find_compartment_index(
    segment_location const& location,
    compartment_model const& graph
);

enum class probeKind {
    membrane_voltage,
    membrane_current
};

struct probe_spec {
    segment_location location;
    probeKind kind;
};

// used in constructor below
struct clone_cell_t {};
constexpr clone_cell_t clone_cell{};

/// high-level abstract representation of a cell and its segments
class cell : public cell_interface {
public:
    using index_type = cell_lid_type;
    using size_type = cell_local_size_type;
    using value_type = double;
    using point_type = point<value_type>;

    struct synapse_instance {
        segment_location location;
        parameter_list mechanism;
    };

    struct stimulus_instance {
        segment_location location;
        i_clamp clamp;
    };

    struct detector_instance {
        segment_location location;
        double threshold;
    };

    // constructor
    cell();

    // sometimes we really do want a copy (pending big morphology
    // refactor)
    cell(clone_cell_t, const cell& other):
        parents_(other.parents_),
        stimuli_(other.stimuli_),
        synapses_(other.synapses_),
        spike_detectors_(other.spike_detectors_),
        probes_(other.probes_)
     {
         // unique_ptr's cannot be copy constructed, do a manual assignment
         segments_.reserve(other.segments_.size());
         for (const auto& s: other.segments_) {
             segments_.push_back(s->clone());
         }
     }

    // Move constructor
    cell(cell&& other) :
        parents_(move(other.parents_)),
        segments_(move(other.segments_)),
        stimuli_(move(other.stimuli_)),
        synapses_(move(other.synapses_)),
        spike_detectors_(move(other.spike_detectors_)),
        probes_(move(other.probes_)) {
        // Set the other resources to default value!
        other.parents_ = std::vector<index_type>();
        other.stimuli_ = std::vector<stimulus_instance>();
        other.synapses_ = std::vector<synapse_instance>();
        other.spike_detectors_ = std::vector<detector_instance>();
        other.probes_ = std::vector<probe_spec>();
        other.segments_ = std::vector<segment_ptr>();
    }

    cell& operator=(cell&& other) {
        if (this != &other) {
            // Nothing to free

            // move all data
            parents_ = move(other.parents_);
            segments_ = move(other.segments_);
            stimuli_ = move(other.stimuli_);
            synapses_ = move(other.synapses_);
            spike_detectors_ = move(other.spike_detectors_);
            probes_ = move(other.probes_);

            // Set the other resources to default value!
            other.parents_ = std::vector<index_type>();
            other.stimuli_ = std::vector<stimulus_instance>();
            other.synapses_ = std::vector<synapse_instance>();
            other.spike_detectors_ = std::vector<detector_instance>();
            other.probes_ = std::vector<probe_spec>();
            other.segments_ = std::vector<segment_ptr>();
        }
        return *this;
    }

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

    std::vector<segment_ptr> const& segments() const;

    /// return reference to array that enumerates the index of the parent of
    /// each segment
    std::vector<index_type> const& segment_parents() const;

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
    void add_synapse(segment_location loc, parameter_list p)
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

    //////////////////
    // probes
    //////////////////
    index_type add_probe(probe_spec p) {
        probes_.push_back(p);
        return probes_.size()-1;
    }

    const std::vector<probe_spec>&
    probes() const { return probes_; }

private:
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

    // the probes
    std::vector<probe_spec> probes_;
};

// Checks that two cells have the same
//  - number and type of segments
//  - volume and area properties of each segment
//  - number of compartments in each segment
bool cell_basic_equality(cell const& lhs, cell const& rhs);

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
cell_description make_cell(const morphology&, bool compartments_from_discretization=false);

} // namespace mc
} // namespace nest
