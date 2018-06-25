#include <arbor/mc_cell.hpp>
#include <arbor/mc_segment.hpp>
#include <arbor/morphology.hpp>

#include "util/rangeutil.hpp"

namespace arb {

using value_type = mc_cell::value_type;
using index_type = mc_cell::index_type;
using size_type = mc_cell::size_type;

mc_cell::mc_cell() {
    // insert a placeholder segment for the soma
    segments_.push_back(make_segment<placeholder_segment>());
    parents_.push_back(0);
}

void mc_cell::assert_valid_segment(index_type i) const {
    if (i>=num_segments()) {
        throw std::out_of_range("no such segment");
    }
}

size_type mc_cell::num_segments() const {
    return segments_.size();
}

//
// note: I think that we have to enforce that the soma is the first
//       segment that is added
//
soma_segment* mc_cell::add_soma(value_type radius, point_type center) {
    if (has_soma()) {
        throw std::runtime_error("cell already has soma");
    }
    segments_[0] = make_segment<soma_segment>(radius, center);
    return segments_[0]->as_soma();
}

cable_segment* mc_cell::add_cable(index_type parent, mc_segment_ptr&& cable) {
    if (!cable->as_cable()) {
        throw std::invalid_argument("segment is not a cable segment");
    }

    if (parent>num_segments()) {
        throw std::out_of_range("parent index out of range");
    }

    segments_.push_back(std::move(cable));
    parents_.push_back(parent);

    return segments_.back()->as_cable();
}

mc_segment* mc_cell::segment(index_type index) {
    assert_valid_segment(index);
    return segments_[index].get();
}

mc_segment const* mc_cell::segment(index_type index) const {
    assert_valid_segment(index);
    return segments_[index].get();
}

bool mc_cell::has_soma() const {
    return !segment(0)->is_placeholder();
}

soma_segment* mc_cell::soma() {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

const soma_segment* mc_cell::soma() const {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

cable_segment* mc_cell::cable(index_type index) {
    assert_valid_segment(index);
    auto cable = segment(index)->as_cable();
    return cable? cable: throw std::runtime_error("segment is not a cable segment");
}

std::vector<size_type> mc_cell::compartment_counts() const {
    std::vector<size_type> comp_count;
    comp_count.reserve(num_segments());
    for (const auto& s: segments()) {
        comp_count.push_back(s->num_compartments());
    }
    return comp_count;
}

size_type mc_cell::num_compartments() const {
    return util::sum_by(segments_,
            [](const mc_segment_ptr& s) { return s->num_compartments(); });
}

void mc_cell::add_stimulus(segment_location loc, i_clamp stim) {
    (void)segment(loc.segment); // assert loc.segment in range
    stimuli_.push_back({loc, std::move(stim)});
}

void mc_cell::add_detector(segment_location loc, double threshold) {
    spike_detectors_.push_back({loc, threshold});
}


// Construct cell from flat morphology specification.

mc_cell make_mc_cell(const morphology& morph, bool compartments_from_discretization) {
    using point3d = mc_cell::point_type;
    mc_cell newcell;

    if (!morph) {
        return newcell;
    }

    arb_assert(morph.check_valid());

    // (not supporting soma-less cells yet)
    newcell.add_soma(morph.soma.r, point3d(morph.soma.x, morph.soma.y, morph.soma.z));

    for (const section_geometry& section: morph.sections) {
        auto kind = section.kind;
        switch (kind) {
        case section_kind::none: // default to dendrite
            kind = section_kind::dendrite;
            break;
        case section_kind::soma:
            throw std::invalid_argument("no support for complex somata");
            break;
        default: ;
        }

        std::vector<value_type> radii;
        std::vector<point3d> points;

        for (const section_point& p: section.points) {
            radii.push_back(p.r);
            points.push_back(point3d(p.x, p.y, p.z));
        }

        auto cable = newcell.add_cable(section.parent_id, kind, radii, points);
        if (compartments_from_discretization) {
            cable->as_cable()->set_compartments(radii.size()-1);
        }
    }

    return newcell;
}

} // namespace arb
