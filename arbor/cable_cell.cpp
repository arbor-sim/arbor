#include <sstream>
#include <unordered_map>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/segment.hpp>
#include <arbor/util/pp_util.hpp>

#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"

namespace arb {

namespace {
    template <typename Property>
    std::string property_key(const Property&) {
        return "";
    }

    std::string property_key(const mechanism_desc& p) {
        return p.name();
    }

    std::string property_key(const initial_ion_data& p) {
        return p.ion;
    }
}

using region_map = std::unordered_map<std::string, mcable_list>;
using locset_map = std::unordered_map<std::string, mlocation_list>;

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

    cable_cell_impl(const arb::morphology& m,
                    const label_dict& dictionary,
                    bool compartments_from_discretization):
        provider(m, dictionary)
    {
        using point = cable_cell::point_type;
        if (!m.num_branches()) {
            segments.push_back(make_segment<placeholder_segment>());
            parents.push_back(0);
            return;
        }

        // Add the soma.
        auto loc = m.samples()[0].loc; // location of soma.

        // If there is no spherical root/soma use a zero-radius soma.
        double srad = m.spherical_root()? loc.radius: 0.;
        segments.push_back(make_segment<soma_segment>(srad, point(loc.x, loc.y, loc.z)));
        parents.push_back(-1);

        auto& samples = m.samples();
        auto& props = m.sample_props();
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
            std::vector<cable_cell::point_type> points;

            // The current discretization code does not handle collocated points correctly,
            // particularly if they lie at the start of a branch, so we have to skip the first
            // point on a branch if it is collocated with the second point.
            bool skip_first = is_collocated(props[index[1]]);
            for (auto j: util::make_span(skip_first, index.size())) {
                auto& s = samples[index[j]];
                radii.push_back(s.loc.radius);
                points.push_back(cable_cell::point_type(s.loc.x, s.loc.y, s.loc.z));
            }

            // Find the id of this branch's parent.
            auto pid = m.branch_parent(i);
            // Adjust pid if a zero-radius soma was used.
            if (!m.spherical_root()) {
                pid = pid==mnpos? 0: pid+1;
            }
            segments.push_back(make_segment<cable_segment>(kind, radii, points));
            parents.push_back(pid);
            if (compartments_from_discretization) {
                int ncolloc = std::count_if(index.begin(), index.end(), [&props](auto i){return is_collocated(props[i]);});
                int ncomp = index.size()-ncolloc-1;
                ncomp -= is_collocated(props[index[0]]);
                segments.back()->as_cable()->set_compartments(ncomp);
            }
        }
    }

    cable_cell_impl(): cable_cell_impl({},{},false) {}

    cable_cell_impl(const cable_cell_impl& other):
        parents(other.parents),
        stimuli(other.stimuli),
        synapses(other.synapses),
        gap_junction_sites(other.gap_junction_sites),
        spike_detectors(other.spike_detectors),
        provider(other.provider)
    {
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
    std::vector<detector_instance> spike_detectors;

    // Embedded morphology and labelled region/locset lookup.
    mprovider provider;

    // Regional assignment.
    cable_cell_region_map region_map;

    // Mechanisms by name and by region.
    // Local ion data by name and by region.

    template <typename Desc, typename T>
    lid_range place(const mlocation_list& locs, const Desc& desc, std::vector<T>& list) {
        const auto first = list.size();

        for (auto loc: locs) {
            list.push_back({loc, desc});
        }

        return lid_range(first, list.size());
    }

    template <typename Desc, typename T>
    lid_range place(const locset& locs, const Desc& desc, std::vector<T>& list) {
        return place(thingify(locs, provider), desc, list);
    }

    lid_range place_gj(const mlocation_list& locs) {
        const auto first = gap_junction_sites.size();

        gap_junction_sites.insert(gap_junction_sites.end(), locs.begin(), locs.end());

        return lid_range(first, gap_junction_sites.size());
    }

    lid_range place_gj(const locset& locs) {
        return place_gj(thingify(locs, provider));
    }

    void assert_valid_segment(index_type i) const {
        if (i>=segments.size()) {
            throw cable_cell_error("no such segment");
        }
    }

    bool valid_location(const mlocation& loc) const {
        return test_invariants(loc) && loc.branch<segments.size();
    }

    void paint_segment(segment_ptr& s, const mechanism_desc& p) {
        s->add_mechanism(p);
    }

    void paint_segment(segment_ptr& s, init_membrane_potential p) {
        s->parameters.init_membrane_potential = p.value;
    }

    void paint_segment(segment_ptr& s, axial_resistivity p) {
        s->parameters.axial_resistivity = p.value;
    }

    void paint_segment(segment_ptr& s, temperature_K p) {
        s->parameters.temperature_K = p.value;
    }

    void paint_segment(segment_ptr& s, membrane_capacitance p) {
        s->parameters.membrane_capacitance = p.value;
    }

    void paint_segment(segment_ptr& s, initial_ion_data p) {
        s->parameters.ion_data[p.ion] = p.initial;
    }

    template <typename Property>
    void paint(const region& reg, const Property& prop) {
        mcable_list cables = thingify(reg, provider);
        auto& mm = region_map.get<Property>()[property_key(prop)];

        for (auto c: cables) {
            if (!mm.insert(c, prop)) {
                throw cable_cell_error(util::pprintf("cable {} overpaints", c));
            }

            if (c.prox_pos!=0 || c.dist_pos!=1) {
                throw cable_cell_error(util::pprintf(
                    "cable_cell does not support regions with partial branches: {}", c));
            }
            assert_valid_segment(c.branch);
            paint_segment(segments[c.branch], prop);
        }
    }
};

using impl_ptr = std::unique_ptr<cable_cell_impl, void (*)(cable_cell_impl*)>;
impl_ptr make_impl(cable_cell_impl* c) {
    return impl_ptr(c, [](cable_cell_impl* p){delete p;});
}

cable_cell::cable_cell(const arb::morphology& m,
                       const label_dict& dictionary,
                       bool compartments_from_discretization):
    impl_(make_impl(new cable_cell_impl(m, dictionary, compartments_from_discretization)))
{}

cable_cell::cable_cell():
    impl_(make_impl(new cable_cell_impl()))
{}

cable_cell::cable_cell(const cable_cell& other):
    default_parameters(other.default_parameters),
    impl_(make_impl(new cable_cell_impl(*other.impl_)))
{}

size_type cable_cell::num_branches() const {
    return impl_->segments.size();
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
    return impl_->spike_detectors;
}

const std::vector<cable_cell::stimulus_instance>& cable_cell::stimuli() const {
    return impl_->stimuli;
}

const concrete_embedding& cable_cell::embedding() const {
    return impl_->provider.embedding();
}

const arb::morphology& cable_cell::morphology() const {
    return impl_->provider.morphology();
}

const mprovider& cable_cell::provider() const {
    return impl_->provider;
}

// Forward paint methods to implementation class.

#define FWD_PAINT(proptype)\
void cable_cell::paint(const region& target, proptype prop) {\
    impl_->paint(target, prop);\
}
ARB_PP_FOREACH(FWD_PAINT,\
    mechanism_desc, init_membrane_potential, axial_resistivity,\
    temperature_K, membrane_capacitance, initial_ion_data)


// Placers.
//
// Implementation of user API for placing discrete items on cell morphology,
// such as synapses, spike detectors and stimuli.

//
// Synapses.
//

lid_range cable_cell::place(const locset& ls, const mechanism_desc& desc) {
    return impl_->place(ls, desc, impl_->synapses);
}

//
// Stimuli.
//

lid_range cable_cell::place(const locset& ls, const i_clamp& desc) {
    return impl_->place(ls, desc, impl_->stimuli);
}

//
// Gap junctions.
//

lid_range cable_cell::place(const locset& ls, gap_junction_site) {
    return impl_->place_gj(ls);
}

//
// Spike detectors.
//

lid_range cable_cell::place(const locset& ls, const threshold_detector& desc) {
    return impl_->place(ls, desc.threshold, impl_->spike_detectors);
}

//
// TODO: deprectate the following as soon as discretization code catches up with em_morphology
//
const soma_segment* cable_cell::soma() const {
    return has_soma()? segment(0)->as_soma(): nullptr;
}

const cable_segment* cable_cell::cable(index_type index) const {
    impl_->assert_valid_segment(index);
    auto cable = segment(index)->as_cable();
    return cable? cable: throw cable_cell_error("segment is not a cable segment");
}

std::vector<size_type> cable_cell::compartment_counts() const {
    std::vector<size_type> comp_count;
    comp_count.reserve(num_branches());
    for (const auto& s: segments()) {
        comp_count.push_back(s->num_compartments());
    }
    return comp_count;
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

} // namespace arb
