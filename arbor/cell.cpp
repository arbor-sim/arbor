#include <cell.hpp>
#include <morphology.hpp>
#include <tree.hpp>
#include <util/rangeutil.hpp>

namespace arb {

cell::cell() {
    // insert a placeholder segment for the soma
    segments_.push_back(make_segment<placeholder_segment>());
    parents_.push_back(0);
}

void cell::assert_valid_segment(index_type i) const {
    if (i>=num_segments()) {
        throw std::out_of_range("no such segment");
    }
}

cell::size_type cell::num_segments() const {
    return segments_.size();
}

//
// note: I think that we have to enforce that the soma is the first
//       segment that is added
//
soma_segment* cell::add_soma(value_type radius, point_type center) {
    if (has_soma()) {
        throw std::runtime_error("cell already has soma");
    }
    segments_[0] = make_segment<soma_segment>(radius, center);
    return segments_[0]->as_soma();
}

cable_segment* cell::add_cable(cell::index_type parent, segment_ptr&& cable) {
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

segment* cell::segment(index_type index) {
    assert_valid_segment(index);
    return segments_[index].get();
}

segment const* cell::segment(index_type index) const {
    assert_valid_segment(index);
    return segments_[index].get();
}

bool cell::has_soma() const {
    return !segment(0)->is_placeholder();
}

soma_segment* cell::soma() {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

const soma_segment* cell::soma() const {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

cable_segment* cell::cable(index_type index) {
    assert_valid_segment(index);
    auto cable = segment(index)->as_cable();
    return cable? cable: throw std::runtime_error("segment is not a cable segment");
}

cell::value_type cell::volume() const {
    return util::sum_by(segments_,
            [](const segment_ptr& s) { return s->volume(); });
}

cell::value_type cell::area() const {
    return util::sum_by(segments_,
            [](const segment_ptr& s) { return s->area(); });
}

std::vector<cell::size_type> cell::compartment_counts() const {
    std::vector<size_type> comp_count;
    comp_count.reserve(num_segments());
    for (const auto& s: segments()) {
        comp_count.push_back(s->num_compartments());
    }
    return comp_count;
}

cell::size_type cell::num_compartments() const {
    return util::sum_by(segments_,
            [](const segment_ptr& s) { return s->num_compartments(); });
}

compartment_model cell::model() const {
    compartment_model m;

    m.tree = tree(parents_);
    auto counts = compartment_counts();
    m.parent_index = make_parent_index(m.tree, counts);
    m.segment_index = algorithms::make_index(counts);

    return m;
}

void cell::add_stimulus(segment_location loc, i_clamp stim) {
    (void)segment(loc.segment); // assert loc.segment in range
    stimuli_.push_back({loc, std::move(stim)});
}

void cell::add_detector(segment_location loc, double threshold) {
    spike_detectors_.push_back({loc, threshold});
}

// Rough and ready comparison of two cells.
// We don't use an operator== because equality of two cells is open to
// interpretation. For example, it is possible to have two viable representations
// of a cell: with and without location information for the cables.
//
// Checks that two cells have the same
//  - number and type of segments
//  - volume and area properties of each segment
//  - number of compartments in each segment
bool cell_basic_equality(const cell& lhs, const cell& rhs) {
    if (lhs.parents_ != rhs.parents_) {
        return false;
    }

    for (cell::index_type i=0; i<lhs.num_segments(); ++i) {
        // a quick and dirty test
        auto& l = *lhs.segment(i);
        auto& r = *rhs.segment(i);

        if (l.kind() != r.kind()) return false;
        if (l.area() != r.area()) return false;
        if (l.volume() != r.volume()) return false;
        if (l.as_cable()) {
            if (l.as_cable()->num_compartments() != r.as_cable()->num_compartments()) {
                return false;
            }
        }
    }
    return true;
}

// Construct cell from flat morphology specification.

cell make_cell(const morphology& morph, bool compartments_from_discretization) {
    using point3d = cell::point_type;
    cell newcell;

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

        std::vector<cell::value_type> radii;
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
