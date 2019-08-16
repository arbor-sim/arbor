#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/segment.hpp>

#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {

using value_type = cable_cell::value_type;
using index_type = cable_cell::index_type;
using size_type = cable_cell::size_type;

cable_cell::cable_cell() {
    // insert a placeholder segment for the soma
    segments_.push_back(make_segment<placeholder_segment>());
    parents_.push_back(0);
}

void cable_cell::assert_valid_segment(index_type i) const {
    if (i>=num_segments()) {
        throw cable_cell_error("no such segment");
    }
}

size_type cable_cell::num_segments() const {
    return segments_.size();
}

//
// note: I think that we have to enforce that the soma is the first
//       segment that is added
//
soma_segment* cable_cell::add_soma(value_type radius, point_type center) {
    if (has_soma()) {
        throw cable_cell_error("cell already has soma");
    }
    segments_[0] = make_segment<soma_segment>(radius, center);
    return segments_[0]->as_soma();
}

cable_segment* cable_cell::add_cable(index_type parent, segment_ptr&& cable) {
    if (!cable->as_cable()) {
        throw cable_cell_error("segment is not a cable segment");
    }

    if (parent>num_segments()) {
        throw cable_cell_error("parent index out of range");
    }

    segments_.push_back(std::move(cable));
    parents_.push_back(parent);

    return segments_.back()->as_cable();
}

segment* cable_cell::segment(index_type index) {
    assert_valid_segment(index);
    return segments_[index].get();
}
segment const* cable_cell::parent(index_type index) const {
    assert_valid_segment(index);
    return segments_[parents_[index]].get();
}

segment const* cable_cell::segment(index_type index) const {
    assert_valid_segment(index);
    return segments_[index].get();
}

bool cable_cell::has_soma() const {
    return !segment(0)->is_placeholder();
}

soma_segment* cable_cell::soma() {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

const soma_segment* cable_cell::soma() const {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

cable_segment* cable_cell::cable(index_type index) {
    assert_valid_segment(index);
    auto cable = segment(index)->as_cable();
    return cable? cable: throw cable_cell_error("segment is not a cable segment");
}

const cable_segment* cable_cell::cable(index_type index) const {
    assert_valid_segment(index);
    auto cable = segment(index)->as_cable();
    return cable? cable: throw cable_cell_error("segment is not a cable segment");
}

std::vector<size_type> cable_cell::compartment_counts() const {
    std::vector<size_type> comp_count;
    comp_count.reserve(num_segments());
    for (const auto& s: segments()) {
        comp_count.push_back(s->num_compartments());
    }
    return comp_count;
}

size_type cable_cell::num_compartments() const {
    return util::sum_by(segments_,
            [](const segment_ptr& s) { return s->num_compartments(); });
}

void cable_cell::add_stimulus(segment_location loc, i_clamp stim) {
    (void)segment(loc.segment); // assert loc.segment in range
    stimuli_.push_back({loc, std::move(stim)});
}

void cable_cell::add_detector(segment_location loc, double threshold) {
    spike_detectors_.push_back({loc, threshold});
}

// Approximating wildly by ignoring O(x) effects entirely, the attenuation b
// over a single cable segment with constant resistivity R and membrane
// capacitance C is given by:
//
// b = 2√(πRCf) · Σ 2L/(√d₀ + √d₁)
//
// where the sum is taken over each piecewise linear segment of length L
// with diameters d₀ and d₁ at each end.

value_type cable_cell::segment_mean_attenuation(
    value_type frequency, index_type segidx,
    const cable_cell_parameter_set& global_defaults) const
{
    value_type R = default_parameters.axial_resistivity.value_or(global_defaults.axial_resistivity.value());
    value_type C = default_parameters.membrane_capacitance.value_or(global_defaults.membrane_capacitance.value());

    value_type length_factor = 0; // [1/√µm]

    if (segidx==0) {
        if (const soma_segment* s = soma()) {
            R = s->parameters.axial_resistivity.value_or(R);
            C = s->parameters.membrane_capacitance.value_or(C);

            value_type d = 2*s->radius();
            length_factor = 1/std::sqrt(d);
        }
    }
    else {
        const cable_segment* s = cable(segidx);
        const auto& lengths = s->lengths();
        const auto& radii = s->radii();

        value_type total_length = 0;
        R = s->parameters.axial_resistivity.value_or(R);
        C = s->parameters.membrane_capacitance.value_or(C);

        for (std::size_t i = 0; i<lengths.size(); ++i) {
            length_factor += 2*lengths[i]/(std::sqrt(radii[i])+std::sqrt(radii[i+1]));
            total_length += lengths[i];
        }
        length_factor /= total_length;
    }

    // R*C is in [s·cm/m²]; need to convert to [s/µm]
    value_type tau_per_um = R*C*1e-8;

    return 2*std::sqrt(math::pi<double>*tau_per_um*frequency)*length_factor; // [1/µm]
}

cable_cell make_cable_cell(const morphology& morph, bool compartments_from_discretization) {
    using point3d = cable_cell::point_type;
    cable_cell newcell;

    if (!morph.num_branches()) {
        return newcell;
    }

    // Add the soma.
    auto loc = morph.samples()[0].loc; // location of soma.

    // If there is no spherical root/soma use a zero-radius soma.
    double srad = morph.spherical_root()? loc.radius: 0.;
    newcell.add_soma(srad, point3d(loc.x, loc.y, loc.z));

    auto& samples = morph.samples();
    for (auto i: util::make_span(1, morph.num_branches())) {
        auto index =  util::make_range(morph.branch_sample_span(i));

        // find kind for the branch. Use the tag of the last sample in the branch.
        int tag = samples[index.back()].tag;
        section_kind kind;
        switch (tag) {
            case 1:     // soma
                throw cable_cell_error("No support for complex somata (yet)");
            case 2:     // axon
                kind = section_kind::axon;
            case 3:     // dendrite
            case 4:     // apical dendrite
            default:    // just take dendrite as default
                kind = section_kind::dendrite;
        }

        std::vector<value_type> radii;
        std::vector<point3d> points;

        for (auto i: index) {
            auto& s = samples[i];
            radii.push_back(s.loc.radius);
            points.push_back(point3d(s.loc.x, s.loc.y, s.loc.z));
        }

        // Find the id of this branch's parent.
        auto pid = morph.branch_parent(i);
        // Adjust pid if a zero-radius soma was used.
        if (!morph.spherical_root()) {
            pid = pid==mnpos? 0: pid+1;
        }
        auto cable = newcell.add_cable(pid, kind, radii, points);
        if (compartments_from_discretization) {
            cable->as_cable()->set_compartments(radii.size()-1);
        }
    }

    return newcell;
}

} // namespace arb
