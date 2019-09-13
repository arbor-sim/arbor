#include <sstream>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/segment.hpp>

#include "morph/em_morphology.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"

namespace arb {

using value_type = cable_cell::value_type;
using index_type = cable_cell::index_type;
using size_type = cable_cell::size_type;

struct cable_cell_impl {
    using value_type = cable_cell::value_type;
    using index_type = cable_cell::index_type;
    using size_type  = cable_cell::size_type;

    using stimulus_instance     = cable_cell::stimulus_instance;
    using synapse_instance      = cable_cell::synapse_instance;
    using gap_junction_instance = cable_cell::gap_junction_instance;
    using detector_instance     = cable_cell::detector_instance;

    using region_map = cable_cell::region_map;
    using locset_map = cable_cell::locset_map;

    cable_cell_impl() {
        segments.push_back(make_segment<placeholder_segment>());
        parents.push_back(0);
    }

    cable_cell_impl(const cable_cell_impl& other) {
        parents = other.parents;
        stimuli = other.stimuli;
        synapses = other.synapses;
        gap_junction_sites = other.gap_junction_sites;
        spike_detectors_ = other.spike_detectors_;
        regions = other.regions;

        // unique_ptr's cannot be copy constructed, do a manual assignment
        segments.reserve(other.segments.size());
        for (const auto& s: other.segments) {
            segments.push_back(s->clone());
        }
    }

    cable_cell_impl(cable_cell_impl&& other) = default;

    // storage for connections
    std::vector<index_type> parents;

    // the segments
    std::vector<segment_ptr> segments;

    // the stimuli
    std::vector<stimulus_instance> stimuli;

    // the synapses
    std::vector<synapse_instance> synapses;

    // the gap_junctions
    std::vector<gap_junction_instance> gap_junction_sites;

    // the sensors
    std::vector<detector_instance> spike_detectors_;

    // Named regions
    region_map regions;

    // Named location sets
    locset_map locations;

    // Underlying embedded morphology
    em_morphology morph;

    void assert_valid_segment(index_type i) const {
        if (i>=segments.size()) {
            throw cable_cell_error("no such segment");
        }
    }
};

using impl_ptr = std::unique_ptr<cable_cell_impl, void (*)(cable_cell_impl*)>;
impl_ptr make_impl(cable_cell_impl* c) {
    return impl_ptr(c, [](cable_cell_impl* p){delete p;});
}

cable_cell::cable_cell():
    impl_(make_impl(new cable_cell_impl()))
{}

cable_cell::cable_cell(const cable_cell& other):
    default_parameters(other.default_parameters),
    impl_(make_impl(new cable_cell_impl(*other.impl_)))
{}

size_type cable_cell::num_segments() const {
    return impl_->segments.size();
}

//
// note: I think that we have to enforce that the soma is the first
//       segment that is added
//
soma_segment* cable_cell::add_soma(value_type radius, point_type center) {
    if (has_soma()) {
        throw cable_cell_error("cell already has soma");
    }
    impl_->segments[0] = make_segment<soma_segment>(radius, center);
    return impl_->segments[0]->as_soma();
}

cable_segment* cable_cell::add_cable(index_type parent, segment_ptr&& cable) {
    if (!cable->as_cable()) {
        throw cable_cell_error("segment is not a cable segment");
    }

    if (parent>num_segments()) {
        throw cable_cell_error("parent index out of range");
    }

    impl_->segments.push_back(std::move(cable));
    impl_->parents.push_back(parent);

    return impl_->segments.back()->as_cable();
}

segment* cable_cell::segment(index_type index) {
    impl_->assert_valid_segment(index);
    return impl_->segments[index].get();
}

segment const* cable_cell::parent(index_type index) const {
    impl_->assert_valid_segment(index);
    return impl_->segments[impl_->parents[index]].get();
}

segment const* cable_cell::segment(index_type index) const {
    impl_->assert_valid_segment(index);
    return impl_->segments[index].get();
}

const std::vector<segment_ptr>& cable_cell::segments() const {
    return impl_->segments;
}

const std::vector<index_type>& cable_cell::parents() const {
    return impl_->parents;
}

value_type cable_cell::segment_length_constant(value_type frequency, index_type segidx,
    const cable_cell_parameter_set& global_defaults) const
{
    return 0.5/segment_mean_attenuation(frequency, segidx, global_defaults);
}

bool cable_cell::has_soma() const {
    return !segment(0)->is_placeholder();
}

const std::vector<cable_cell::gap_junction_instance>& cable_cell::gap_junction_sites() const {
    return impl_->gap_junction_sites;
}

const std::vector<cable_cell::synapse_instance>& cable_cell::synapses() const {
    return impl_->synapses;
}

const std::vector<cable_cell::detector_instance>& cable_cell::detectors() const {
    return impl_->spike_detectors_;
}

const std::vector<cable_cell::stimulus_instance>& cable_cell::stimuli() const {
    return impl_->stimuli;
}

void cable_cell::set_regions(cable_cell::region_map r) {
    impl_->regions = std::move(r);
}

void cable_cell::set_locsets(cable_cell::locset_map l) {
    impl_->locations = std::move(l);
}

//
// Painters.
//
// Implementation of user API for painting density channel and electrical properties on cells.
//

void cable_cell::paint(const std::string& target, mechanism_desc desc) {
    auto it = impl_->regions.find(target);

    // Nothing to do if there are no regions that match.
    if (it==impl_->regions.end()) return;

    for (auto c: it->second) {
        if (c.prox_pos!=0 || c.dist_pos!=1) {
            throw cable_cell_error(util::pprintf(
                "cable_cell does not support regions with partial branches: \"{}\": {}",
                target, c));
        }
        segment(c.branch)->add_mechanism(std::move(desc));
    }
}

//
// Placers.
//
// Implementation of user API for placing discrete items on cell morphology,
// such as synapses, spike detectors and stimuli.
//

//
// Synapses
//

locrange cable_cell::place(const std::string& target, mechanism_desc desc) {
    auto first = impl_->synapses.size();

    auto it = impl_->locations.find(target);
    if (it==impl_->locations.end()) return locrange(first, first);

    impl_->synapses.reserve(first+it->second.size());
    for (auto loc: it->second) {
        impl_->synapses.push_back({loc, desc});
    }

    return locrange(first, impl_->synapses.size());
}

locrange cable_cell::place(mlocation loc, mechanism_desc desc) {
    if (!test_invariants(loc) || loc.branch>=num_segments()) {
        throw cable_cell_error(util::pprintf(
            "Attempt to add synapse at invalid location: \"{}\"", loc));
    }
    auto first = impl_->synapses.size();
    impl_->synapses.push_back({loc, desc});
    return locrange(first, first+1);
}

//
// Stimuli
//

locrange cable_cell::place(const std::string& target, i_clamp desc) {
    auto first = impl_->stimuli.size();

    auto it = impl_->locations.find(target);
    if (it==impl_->locations.end()) return locrange(first, first);

    impl_->stimuli.reserve(first+it->second.size());
    for (auto loc: it->second) {
        impl_->stimuli.push_back({loc, desc});
    }

    return locrange(first, impl_->stimuli.size());
}

locrange cable_cell::place(mlocation loc, i_clamp stim) {
    if (!test_invariants(loc) || loc.branch>=num_segments()) {
        throw cable_cell_error(util::pprintf(
            "Attempt to add stimulus at invalid location: \"{}\"", loc));
    }
    auto first = impl_->stimuli.size();
    impl_->stimuli.push_back({loc, std::move(stim)});
    return locrange(first, first+1);
}

//
// Gap junctions.
//

locrange cable_cell::place(const std::string& target, gap_junction_site) {
    auto first = impl_->gap_junction_sites.size();

    auto it = impl_->locations.find(target);
    if (it==impl_->locations.end()) return locrange(first, first);

    impl_->gap_junction_sites.reserve(first+it->second.size());
    for (auto loc: it->second) {
        impl_->gap_junction_sites.push_back(loc);
    }

    return locrange(first, impl_->gap_junction_sites.size());
}

locrange cable_cell::place(mlocation loc, gap_junction_site) {
    if (!test_invariants(loc) || loc.branch>=num_segments()) {
        throw cable_cell_error(util::pprintf(
            "Attempt to add gap junction at invalid location: \"{}\"", loc));
    }
    auto first = impl_->gap_junction_sites.size();
    impl_->gap_junction_sites.push_back(loc);
    return locrange(first, first+1);
}

//
// Spike detectors.
//

locrange cable_cell::place(mlocation loc, detector d) {
    if (!test_invariants(loc) || loc.branch>=num_segments()) {
        throw cable_cell_error(util::pprintf(
            "Attempt to add spike detector at invalid location: \"{}\"", loc));
    }
    auto first = impl_->spike_detectors_.size();
    impl_->spike_detectors_.push_back({loc, d.threshold});
    return locrange(first, first+1);
}


soma_segment* cable_cell::soma() {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

const soma_segment* cable_cell::soma() const {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

cable_segment* cable_cell::cable(index_type index) {
    impl_->assert_valid_segment(index);
    auto cable = segment(index)->as_cable();
    return cable? cable: throw cable_cell_error("segment is not a cable segment");
}

const cable_segment* cable_cell::cable(index_type index) const {
    impl_->assert_valid_segment(index);
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

const em_morphology* cable_cell::morphology() const {
    return &(impl_->morph);
}

void cable_cell::set_morphology(em_morphology m) {
    impl_->morph = std::move(m);
}

size_type cable_cell::num_compartments() const {
    return util::sum_by(impl_->segments,
            [](const segment_ptr& s) { return s->num_compartments(); });
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
    value_type R = default_parameters.axial_resistivity.value_or(
            global_defaults.axial_resistivity.value());
    value_type C = default_parameters.membrane_capacitance.value_or(
            global_defaults.membrane_capacitance.value());

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

cable_cell make_cable_cell(const morphology& m,
                           const label_dict& dictionary,
                           bool compartments_from_discretization)
{
    using point3d = cable_cell::point_type;
    cable_cell newcell;

    if (!m.num_branches()) {
        return newcell;
    }

    // Add the soma.
    auto loc = m.samples()[0].loc; // location of soma.

    // If there is no spherical root/soma use a zero-radius soma.
    double srad = m.spherical_root()? loc.radius: 0.;
    newcell.add_soma(srad, point3d(loc.x, loc.y, loc.z));

    auto& samples = m.samples();
    for (auto i: util::make_span(1, m.num_branches())) {
        auto index =  util::make_range(m.branch_indexes(i));

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
        auto pid = m.branch_parent(i);
        // Adjust pid if a zero-radius soma was used.
        if (!m.spherical_root()) {
            pid = pid==mnpos? 0: pid+1;
        }
        auto cable = newcell.add_cable(pid, make_segment<cable_segment>(kind, radii, points));
        if (compartments_from_discretization) {
            cable->as_cable()->set_compartments(radii.size()-1);
        }
    }

    // Construct concrete regions.
    // Ignores the pointsets, for now.
    auto em = em_morphology(m); // for converting labels to "segments"

    std::unordered_map<std::string, mcable_list> regions;
    for (auto r: dictionary.regions()) {
        regions[r.first] = thingify(r.second, em);
    }
    newcell.set_regions(std::move(regions));

    std::unordered_map<std::string, mlocation_list> locsets;
    for (auto l: dictionary.locsets()) {
        locsets[l.first] = thingify(l.second, em);
    }
    newcell.set_locsets(std::move(locsets));

    return newcell;
}

} // namespace arb
