#pragma once

#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "segment.hpp"
#include "cell_tree.hpp"
#include "stimulus.hpp"

namespace nest {
namespace mc {

/// wrapper around compartment layout information derived from a high level cell
/// description
struct compartment_model {
    cell_tree tree;
    std::vector<int> parent_index;
    std::vector<int> segment_index;
};

struct segment_location {
    segment_location(int s, double l)
    : segment(s), position(l)
    {
        EXPECTS(position>=0. && position<=1.);
    }
    friend bool operator==(segment_location l, segment_location r) {
        return l.segment==r.segment && l.position==r.position;
    }
    int segment;
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

/// high-level abstract representation of a cell and its segments
class cell {
public:

    // types
    using index_type = int;
    using value_type = double;
    using point_type = point<value_type>;
    struct probe_instance {
        segment_location location;
        probeKind kind;
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
    cable_segment* add_cable(index_type parent, Args ...args);

    /// the number of segments in the cell
    int num_segments() const;

    bool has_soma() const;

    class segment* segment(int index);
    class segment const* segment(int index) const;

    /// access pointer to the soma
    /// returns nullptr if the cell has no soma
    soma_segment* soma();

    /// access pointer to a cable segment
    /// will throw an std::out_of_range exception if
    /// the cable index is not valid
    cable_segment* cable(int index);

    /// the volume of the cell
    value_type volume() const;

    /// the surface area of the cell
    value_type area() const;

    /// the total number of compartments over all segments
    size_t num_compartments() const;

    std::vector<segment_ptr> const& segments() const;

    /// return reference to array that enumerates the index of the parent of
    /// each segment
    std::vector<int> const& segment_parents() const;

    /// return a vector with the compartment count for each segment in the cell
    std::vector<int> compartment_counts() const;

    compartment_model model() const;

    //////////////////
    // stimulii
    //////////////////
    void add_stimulus(segment_location loc, i_clamp stim);

    std::vector<stimulus_instance>&
    stimulii() {
        return stimulii_;
    }

    const std::vector<stimulus_instance>&
    stimulii() const {
        return stimulii_;
    }

    //////////////////
    // synapses
    //////////////////
    void add_synapse(segment_location loc);

    const std::vector<segment_location>& synapses() const;

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
    index_type add_probe(segment_location loc, probeKind kind) {
        probes_.push_back({loc, kind});
        return probes_.size()-1;
    }

    const std::vector<probe_instance>&
    probes() const { return probes_; }

private:

    // storage for connections
    std::vector<index_type> parents_;

    // the segments
    std::vector<segment_ptr> segments_;

    // the stimulii
    std::vector<stimulus_instance> stimulii_;

    // the synapses
    std::vector<segment_location> synapses_;

    // the sensors
    std::vector<detector_instance> spike_detectors_;

    // the probes
    std::vector<probe_instance> probes_;
};

// Checks that two cells have the same
//  - number and type of segments
//  - volume and area properties of each segment
//  - number of compartments in each segment
bool cell_basic_equality(cell const& lhs, cell const& rhs);

// create a cable by forwarding cable construction parameters provided by the user
template <typename... Args>
cable_segment* cell::add_cable(cell::index_type parent, Args ...args)
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

} // namespace mc
} // namespace nest

