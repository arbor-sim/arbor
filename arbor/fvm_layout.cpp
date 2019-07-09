#include <set>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>

#include "algorithms.hpp"
#include "fvm_compartment.hpp"
#include "fvm_layout.hpp"
#include "tree.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/transform.hpp"

namespace arb {

using util::count_along;
using util::make_span;
using util::subrange_view;
using util::transform_view;
using util::value_by_key;

// Convenience routines

namespace {
    template <typename ResizableContainer, typename Index>
    void extend_to(ResizableContainer& c, const Index& i) {
        if (util::size(c)<=i) {
            c.resize(i+1);
        }
    }

    struct compartment_model {
        arb::tree tree;
        std::vector<tree::int_type> parent_index;
        std::vector<tree::int_type> segment_index;

        explicit compartment_model(const cable_cell& cell) {
            tree = arb::tree(cell.parents());
            auto counts = cell.compartment_counts();
            for (unsigned i = 0; i < cell.segments().size(); i++) {
                if (!cell.segment(i)->is_soma() && cell.parent(i)->is_soma()) {
                    counts[i]++;
                }
            }
            parent_index = make_parent_index(tree, counts);
            segment_index = algorithms::make_index(counts);
        }
    };

    struct cv_param {
        fvm_size_type cv;
        std::vector<fvm_value_type> params;
        fvm_size_type target;
    };

    ARB_DEFINE_LEXICOGRAPHIC_ORDERING(cv_param,(a.cv,a.params,a.target),(b.cv,b.params,b.target))
} // namespace

// Cable segment discretization
// ----------------------------
//
// Each compartment i straddles the ith control volume on the right
// and the jth control volume on the left, where j is the parent index
// of i.
//
// Dividing the comparment into two halves, the centre face C
// corresponds to the shared face between the two control volumes,
// the surface areas in each half contribute to the surface area of
// the respective control volumes, and the volumes and lengths of
// each half are used to calculate the flux coefficients that
// for the connection between the two control volumes and which
// is stored in `face_conductance[i]`.
//
//
//  +------- cv j --------+------- cv i -------+
//  |                     |                    |
//  v                     v                    v
//  ____________________________________________
//  | ........ | ........ |          |         |
//  | ........ L ........ C          R         |
//  |__________|__________|__________|_________|
//             ^                     ^
//             |                     |
//             +--- compartment i ---+
//
// The first control volume of any cell corresponds to the soma
// and the first half of the first cable compartment of that cell.
//
//
// Face conductance computation
// ----------------------------
//
// The conductance between two adjacent CVs is computed as follows,
// computed in terms of the two half CVs on either side of the interface,
// correspond to the regions L–C and C–R in the diagram above.
//
// The conductance itself is approximated by the weighted harmonic mean
// of the mean linear conductivities in each half, corresponding to
// the two-point flux approximation in 1-D.
//
// Mean linear conductivities:
//
//     g₁ = 1/h₁ ∫₁ A(x)/R dx
//     g₂ = 1/h₂ ∫₂ A(x)/R dx
//
// where A(x) is the cross-sectional area, R is the bulk resistivity,
// and h is the width of the region. The integrals are taken over the
// half CVs as described above.
//
// Equivalently, in terms of the semi-compartment volumes V₁ and V₂:
//
//     g₁ = 1/R·V₁/h₁
//     g₂ = 1/R·V₂/h₂
//
// Weighted harmonic mean, with h = h₁+h₂:
//
//     g = (h₁/h·g₁¯¹+h₂/h·g₂¯¹)¯¹
//       = 1/R · hV₁V₂/(h₂²V₁+h₁²V₂)
//

fvm_discretization fvm_discretize(const std::vector<cable_cell>& cells) {

    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type = fvm_size_type;

    fvm_discretization D;

    util::make_partition(D.cell_segment_bounds,
        transform_view(cells, [](const cable_cell& c) { return c.num_segments(); }));

    std::vector<index_type> cell_cv_bounds;
    auto cell_cv_part = make_partition(cell_cv_bounds,
        transform_view(cells, [](const cable_cell& c) {
            unsigned ncv = 0;
            for (unsigned i = 0; i < c.segments().size(); i++) {
                ncv += c.segment(i)->num_compartments();
                if (!c.segment(i)->is_soma() && c.parent(i)->is_soma()) {
                    ncv++;
                }
            }
            return ncv;
        }));

    D.ncell = cells.size();
    D.ncv = cell_cv_part.bounds().second;

    D.face_conductance.assign(D.ncv, 0.);
    D.cv_area.assign(D.ncv, 0.);
    D.cv_capacitance.assign(D.ncv, 0.);
    D.parent_cv.assign(D.ncv, index_type(-1));
    D.cv_to_cell.resize(D.ncv);
    for (auto i: make_span(0, D.ncell)) {
        util::fill(subrange_view(D.cv_to_cell, cell_cv_part[i]), static_cast<index_type>(i));
    }

    std::vector<size_type> seg_cv_bounds;
    for (auto i: make_span(0, D.ncell)) {
        const auto& c = cells[i];
        compartment_model cell_graph(c);
        auto cell_cv_ival = cell_cv_part[i];

        auto cell_cv_base = cell_cv_ival.first;
        for (auto k: make_span(cell_cv_ival)) {
            D.parent_cv[k] = cell_graph.parent_index[k-cell_cv_base]+cell_cv_base;
        }

        // Compartment index range for each segment in this cell.
        seg_cv_bounds.clear();
        auto seg_cv_part = make_partition(
            seg_cv_bounds,
            transform_view(make_span(c.num_segments()), [&c](const unsigned s) {
                if (!c.segment(s)->is_soma() && c.parent(s)->is_soma()) {
                    return c.segment(s)->num_compartments() + 1;
                }
                return c.segment(s)->num_compartments();
            }),
            cell_cv_base);

        const auto nseg = seg_cv_part.size();
        if (nseg==0) {
            throw arbor_internal_error("fvm_layout: cannot discretrize cell with no segments");
        }

        // Handle soma (first segment and root of tree) specifically.
        const auto soma = c.segment(0)->as_soma();
        if (!soma) {
            throw arbor_internal_error("fvm_layout: first segment of cell must be soma");
        }
        else if (soma->num_compartments()!=1) {
            throw arbor_internal_error("fvm_layout: soma must have exactly one compartment");
        }

        segment_info soma_info;

        size_type soma_cv = cell_cv_base;
        value_type soma_area = math::area_sphere(soma->radius());

        soma_info.proximal_cv = soma_cv;
        soma_info.distal_cv = soma_cv;
        soma_info.distal_cv_area = soma_area;
        D.segments.push_back(soma_info);

        // Other segments must all be cable segments.
        for (size_type j = 1; j<nseg; ++j) {
            const auto& seg_cv_ival = seg_cv_part[j];
            const auto ncv = seg_cv_ival.second-seg_cv_ival.first;

            segment_info seg_info;

            const auto cable = c.segment(j)->as_cable();
            if (!cable) {
                throw arbor_internal_error("fvm_layout: non-root segments of cell must be cable segments");
            }
            auto cm = cable->cm;    // [F/m²]
            auto rL = cable->rL;    // [Ω·cm]

            bool soma_parent = c.parent(j)->as_soma() ? true : false; //segment's parent is a soma

            auto radii = cable->radii();
            auto lengths = cable->lengths();

            // If segment has soma parent, send soma information to div_compartment_integrator
            if(soma_parent) {
                radii.insert(radii.begin(), soma->radius());
                lengths.insert(lengths.begin(), soma->radius()*2);
            }

            auto divs = div_compartment_integrator(ncv, radii, lengths, soma_parent);

            seg_info.parent_cv = D.parent_cv[seg_cv_ival.first];
            seg_info.parent_cv_area = soma_parent ? 0 : divs(0).left.area;
            seg_info.soma_parent = soma_parent;

            seg_info.proximal_cv = seg_cv_ival.first;
            seg_info.distal_cv = seg_cv_ival.second-1;
            seg_info.distal_cv_area = divs(ncv-1).right.area;

            D.segments.push_back(seg_info);

            for (auto i: make_span(seg_cv_ival)) {
                const auto& div = divs(i-seg_cv_ival.first);
                auto j = D.parent_cv[i];

                auto h1 = div.left.length;       // [µm]
                auto V1 = div.left.volume;       // [µm³]
                auto h2 = div.right.length;      // [µm]
                auto V2 = div.right.volume;      // [µm³]
                auto h = h1+h2;

                auto linear_conductivity = 1/rL*h*V1*V2/(h2*h2*V1+h1*h1*V2); // [S·cm¯¹·µm²] ≡ [10²·µS·µm]
                constexpr double unit_scale = 1e2;
                D.face_conductance[i] =  unit_scale * linear_conductivity / h; // [µS]

                auto al = div.left.area;         // [µm²]
                auto ar = div.right.area;        // [µm²]

                D.cv_area[j] += al;              // [µm²]
                D.cv_capacitance[j] += al * cm;  // [pF]
                D.cv_area[i] += ar;              // [µm²]
                D.cv_capacitance[i] += ar * cm;  // [pF]
            }
        }

        D.cv_area[soma_cv] = soma_area;                  // [µm²]
        D.cv_capacitance[soma_cv] = soma_area*soma->cm;  // [pF]
    }

    // Number of CVs per cell is exactly number of compartments.
    D.cell_cv_bounds = std::move(cell_cv_bounds);
    return D;
}

// Build up mechanisms.
//
// Processing procedes in the following stages:
//
//   I.  Collect segment mechanism info from the cell descriptions into temporary
//       data structures for density mechanism, point mechanisms, and ion channels.
//
//   II. Build mechanism and ion configuration in `fvm_mechanism_data`:
//       IIa. Ion channel CVs.
//       IIb. Density mechanism CVs, parameter values; ion channel default concentration contributions.
//       IIc. Point mechanism CVs, parameter values, and targets.

fvm_mechanism_data fvm_build_mechanism_data(const cable_cell_global_properties& gprop, const std::vector<cable_cell>& cells, const fvm_discretization& D) {
    using util::assign;
    using util::sort_by;
    using util::optional;

    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type = fvm_size_type;

    using string_set = std::unordered_set<std::string>;
    using string_index_map = std::unordered_map<std::string, size_type>;

    const mechanism_catalogue& catalogue = *gprop.catalogue;

    fvm_mechanism_data mechdata;

    // I. Collect segment mechanism info from cells.

    // Temporary table for density mechanism info, mapping mechanism name to tuple of:
    //     1. Vector of segment indices and mechanism parameter settings where mechanism occurs.
    //     2. Set of the names of parameters that are anywhere modified.
    //     3. Pointer to mechanism metadata from catalogue.

    struct density_mech_data {
        std::vector<std::pair<size_type, const mechanism_desc*>> segments; //
        string_set paramset;
        const mechanism_info* info = nullptr;
    };
    std::unordered_map<std::string, density_mech_data> density_mech_table;

    // Temporary table for point mechanism info, mapping mechanism name to tuple:
    //     1. Vector of point info: CV, index into cell group targets, parameter settings.
    //     2. Set of the names of parameters that are anywhere modified.
    //     3. Mechanism parameter settings.

    struct point_mech_data {
        struct point_data {
            size_type cv;
            size_type target_index;
            const mechanism_desc* desc;
        };
        std::vector<point_data> points;
        string_set paramset;
        const mechanism_info* info = nullptr;
    };
    std::unordered_map<std::string, point_mech_data> point_mech_table;

    // Built-in stimulus mechanism data is dealt with especially below.
    // Record for each stimulus the CV and clamp data.
    std::vector<std::pair<size_type, i_clamp>> stimuli;

    // Temporary table for presence of ion channels, mapping ion name to _sorted_
    // collection of segment indices.

    std::unordered_map<std::string, std::set<size_type>> ion_segments;

    auto update_paramset_and_validate =
        [&catalogue]
        (const mechanism_desc& desc, const mechanism_info*& info, string_set& paramset)
    {
        auto& name = desc.name();
        if (!info) {
            info = &catalogue[name];
        }
        for (const auto& pv: desc.values()) {
            if (!paramset.count(pv.first)) {
                if (!info->parameters.count(pv.first)) {
                    throw no_such_parameter(name, pv.first);
                }
                if (!info->parameters.at(pv.first).valid(pv.second)) {
                    throw invalid_parameter_value(name, pv.first, pv.second);
                }
                paramset.insert(pv.first);
            }
        }
    };

    auto verify_ion_usage =
        [&gprop](const std::string& mech_name, const mechanism_info* info)
    {
        const auto& global_ions = gprop.ion_default;

        for (const auto& ion: info->ions) {
            const auto& ion_name = ion.first;
            const auto& ion_dep = ion.second;

            if (!global_ions.count(ion_name)) {
                throw cable_cell_error(
                    "mechanism "+mech_name+" uses ion "+ion_name+ " which is missing in global properties");
            }

            if (ion_dep.verify_ion_charge) {
                if (ion_dep.expected_ion_charge!=global_ions.at(ion_name).charge) {
                    throw cable_cell_error(
                        "mechanism "+mech_name+" uses ion "+ion_name+ " expecting a different valence");
                }
            }
        }
    };

    auto cell_segment_part = D.cell_segment_part();
    size_type target_id = 0;

    for (auto cell_idx: make_span(0, D.ncell)) {
        auto& cell = cells[cell_idx];
        auto seg_range = cell_segment_part[cell_idx];

        for (auto segment_idx: make_span(seg_range)) {
            for (const mechanism_desc& desc: cell.segments()[segment_idx-seg_range.first]->mechanisms()) {
                const auto& name = desc.name();

                density_mech_data& entry = density_mech_table[name];
                update_paramset_and_validate(desc, entry.info, entry.paramset);
                entry.segments.emplace_back(segment_idx, &desc);

                for (const auto& ion: entry.info->ions) {
                    ion_segments[ion.first].insert(segment_idx);
                }
            }
        }

        for (const auto& cellsyn: cell.synapses()) {
            const mechanism_desc& desc = cellsyn.mechanism;
            size_type cv = D.segment_location_cv(cell_idx, cellsyn.location);
            const auto& name = desc.name();

            point_mech_data& entry = point_mech_table[name];
            update_paramset_and_validate(desc, entry.info, entry.paramset);
            entry.points.push_back({cv, target_id++, &desc});

            size_type segment_idx = D.cell_segment_bounds[cell_idx]+cellsyn.location.segment;
            for (const auto& ion: entry.info->ions) {
                ion_segments[ion.first].insert(segment_idx);
            }
        }

        for (const auto& stimulus: cell.stimuli()) {
            size_type cv = D.segment_location_cv(cell_idx, stimulus.location);
            stimuli.push_back({cv, stimulus.clamp});
        }
    }

    for (auto& entry: density_mech_table) {
        verify_ion_usage(entry.first, entry.second.info);
    }

    for (auto& entry: point_mech_table) {
        verify_ion_usage(entry.first, entry.second.info);
    }

    // II. Build ion and mechanism configs.

    // Shared temporary lookup info across mechanism instances, set by build_param_data.
    string_index_map param_index;
    std::vector<std::string> param_name;
    std::vector<value_type> param_default;

    auto build_param_data =
        [&param_name, &param_index, &param_default](const string_set& paramset, const mechanism_info* info)
    {
        assign(param_name, paramset);
        auto nparam = paramset.size();

        assign(param_default, transform_view(param_name,
            [info](const std::string& p) { return info->parameters.at(p).default_value; }));

        param_index.clear();
        for (auto i: make_span(0, nparam)) {
            param_index[param_name[i]] = i;
        }
        return nparam;
    };

    // IIa. Ion channel CVs.

    for (auto& ionseg: ion_segments) {
        auto& ion = mechdata.ions[ionseg.first];

        for (size_type segment: ionseg.second) {
            const segment_info& seg_info = D.segments[segment];

            if (seg_info.has_parent()) {
                index_type cv = seg_info.parent_cv;
                optional<std::size_t> parent_idx = util::binary_search_index(ion.cv, cv);
                if (!parent_idx) {
                    ion.cv.push_back(cv);
                    ion.iconc_norm_area.push_back(D.cv_area[cv]);
                    ion.econc_norm_area.push_back(D.cv_area[cv]);
                }
            }

            for (auto cv: make_span(seg_info.cv_range())) {
                ion.cv.push_back(cv);
                ion.iconc_norm_area.push_back(D.cv_area[cv]);
                ion.econc_norm_area.push_back(D.cv_area[cv]);
            }
        }
    }

    // IIb. Density mechanism CVs, parameters and ionic default concentration contributions.

    // Ameliorate area sum rounding areas by clamping normalized area contributions to [0, 1]
    // and rounding values within an epsilon of 0 or 1 to that value.
    auto trim = [](value_type& v) {
        constexpr value_type eps = std::numeric_limits<value_type>::epsilon()*4;
        v = v<eps? 0: v+eps>1? 1: v;
    };

    for (const auto& entry: density_mech_table) {
        const std::string& name = entry.first;
        fvm_mechanism_config& config = mechdata.mechanisms[name];
        config.kind = mechanismKind::density;

        auto nparam = build_param_data(entry.second.paramset, entry.second.info);

        // In order to properly account for partially overriden paramaters in CVs
        // that are shared between segments, we need to track not only the area-weighted
        // sum of parameter values, but also the total area for each CV for each parameter
        // that has been overriden — the remaining area demands a contribution from the
        // parameter default value.

        std::vector<std::vector<value_type>> param_value(nparam);
        std::vector<std::vector<value_type>> param_area_contrib(nparam);

        // (gcc 6.x bug fails to deduce const in lambda capture reference initialization)
        const auto& info = *entry.second.info;
        auto accumulate_mech_data =
            [
                &info,
                &ion_configs = mechdata.ions,
                &param_index, &param_value, &param_area_contrib, &config
            ]
            (size_type index, index_type cv, value_type area, const mechanism_desc& desc)
        {
            for (auto& kv: desc.values()) {
                int pidx = param_index.at(kv.first);
                value_type v = kv.second;

                extend_to(param_area_contrib[pidx], index);
                param_area_contrib[pidx][index] += area;

                extend_to(param_value[pidx], index);
                param_value[pidx][index] += area*v;
            }

            for (auto& ion: info.ions) {
                fvm_ion_config& ion_config = ion_configs[ion.first];
                size_type index = util::binary_search_index(ion_config.cv, cv).value();
                if (ion.second.write_concentration_int) {
                    ion_config.iconc_norm_area[index] -= area;
                }
                if (ion.second.write_concentration_ext) {
                    ion_config.econc_norm_area[index] -= area;
                }
            }

            extend_to(config.norm_area, index);
            config.norm_area[index] += area;
        };

        for (auto& seg_entry: entry.second.segments) {
            const segment_info& seg_info = D.segments[seg_entry.first];
            const mechanism_desc& mech_desc = *seg_entry.second;

            if (seg_info.has_parent()) {
                index_type cv = seg_info.parent_cv;
                optional<std::size_t> parent_idx = util::binary_search_index(config.cv, cv);
                if (!parent_idx) {
                    parent_idx = config.cv.size();
                    config.cv.push_back(cv);
                }

                accumulate_mech_data(*parent_idx, cv, seg_info.parent_cv_area, mech_desc);
            }

            for (auto cv: make_span(seg_info.cv_range())) {
                size_type idx = config.cv.size();
                config.cv.push_back(cv);

                value_type area = cv==seg_info.distal_cv? seg_info.distal_cv_area: D.cv_area[cv];
                accumulate_mech_data(idx, cv, area, mech_desc);
            }
        }

        // Complete parameter values with default values.

        config.param_values.resize(nparam);
        for (auto pidx: make_span(0, nparam)) {
            value_type default_value = param_default[pidx];
            config.param_values[pidx].first = param_name[pidx];

            auto& values = config.param_values[pidx].second;
            values.resize(config.cv.size());

            for (auto i: count_along(config.cv)) {
                value_type v = param_value[pidx][i];
                value_type cv_area = D.cv_area[config.cv[i]];
                value_type remaining_area = cv_area-param_area_contrib[pidx][i];

                values[i] = (v+remaining_area*default_value)/cv_area;
            }
        }

        // Normalize norm_area entries.

        for (auto i: count_along(config.cv)) {
            config.norm_area[i] /= D.cv_area[config.cv[i]];
            trim(config.norm_area[i]);
        }
    }

    // Normalize ion norm_area entries.

    for (auto& entry: mechdata.ions) {
        auto& ion_config = entry.second;
        for (auto i: count_along(ion_config.cv)) {
            auto cv_area = D.cv_area[ion_config.cv[i]];
            ion_config.iconc_norm_area[i] /= cv_area;
            trim(ion_config.iconc_norm_area[i]);

            ion_config.econc_norm_area[i] /= cv_area;
            trim(ion_config.econc_norm_area[i]);
        }
    }

    // II.3 Point mechanism CVs, targets, parameters and stimuli.

    for (const auto& entry: point_mech_table) {
        const std::string& name = entry.first;
        const auto& points = entry.second.points;

        auto nparam = build_param_data(entry.second.paramset, entry.second.info);
        std::vector<std::vector<value_type>> param_value(nparam);

        // Permute points in this mechanism so that they are in increasing CV order;
        // cv_order[i] is the index of the ith point by increasing CV.

        mechdata.ntarget += points.size();

        std::vector<size_type> cv_order;
        assign(cv_order, count_along(points));
        sort_by(cv_order, [&](size_type i) { return points[i].cv; });

        fvm_mechanism_config& config = mechdata.mechanisms[name];
        config.kind = mechanismKind::point;

        // Generate config.cv: contains cv of group of synapses that can be coalesced into one instance
        // Generate config.param_values: contains parameters of group of synapses that can be coalesced into one instance
        // Generate multiplicity: contains number of synapses in each coalesced group of synapses
        // Generate target: contains the synapse target number
        if(catalogue[name].linear && gprop.coalesce_synapses) {
            // cv_param_vec used to lexicographically sort the cv, parameters and target, which are stored in that order
            std::vector<cv_param> cv_param_vec(cv_order.size());

            for (unsigned i = 0; i < cv_order.size(); ++i) {
                auto loc = cv_order[i];
                std::vector<value_type> p(nparam);
                for (auto pidx: make_span(0, nparam)) {
                    value_type pdefault = param_default[pidx];
                    const std::string& pname = param_name[pidx];
                    p[pidx] = value_by_key(points[loc].desc->values(), pname).value_or(pdefault);
                }
                cv_param_vec[i] = cv_param{points[loc].cv, p, points[loc].target_index};
            }

            std::sort(cv_param_vec.begin(), cv_param_vec.end());

            auto identical_synapse = [](const cv_param& i, const cv_param& j) {
                return i.cv==j.cv && i.params==j.params;
            };

            config.param_values.resize(nparam);
            for (auto pidx: make_span(0, nparam)) {
                config.param_values[pidx].first = param_name[pidx];
            }
            config.target.reserve(cv_param_vec.size());

            for (auto i = cv_param_vec.begin(), j = i; i!=cv_param_vec.end(); i = j) {
                ++j;
                while (j!=cv_param_vec.end() && identical_synapse(*i, *j)) ++j;

                auto mergeable = util::make_range(i, j);

                config.cv.push_back((*i).cv);
                for (auto pidx: make_span(0, nparam)) {
                    config.param_values[pidx].second.push_back((*i).params[pidx]);
                }
                config.multiplicity.push_back(mergeable.size());

                for (auto e: mergeable) {
                    config.target.push_back(e.target);
                }
            }
        }
        else {
            assign(config.cv, transform_view(cv_order, [&](size_type j) { return points[j].cv; }));
            assign(config.target, transform_view(cv_order, [&](size_type j) { return points[j].target_index; }));

            config.param_values.resize(nparam);
            for (auto pidx: make_span(0, nparam)) {
                value_type pdefault = param_default[pidx];
                const std::string& pname = param_name[pidx];

                config.param_values[pidx].first = pname;

                auto& values = config.param_values[pidx].second;
                assign(values, transform_view(cv_order,
                                              [&](size_type j) { return value_by_key(points[j].desc->values(), pname).value_or(pdefault); }));
            }
        }
    }

    // Sort stimuli by ascending CV and construct parameter vectors.
    if (!stimuli.empty()) {
        fvm_mechanism_config& stim_config = mechdata.mechanisms["_builtin_stimulus"];
        using cv_clamp = const std::pair<size_type, i_clamp>&;

        auto stim_cv_field = [](cv_clamp p) { return p.first; };
        sort_by(stimuli, stim_cv_field);
        assign(stim_config.cv, transform_view(stimuli, stim_cv_field));

        stim_config.param_values.resize(3);

        stim_config.param_values[0].first = "delay";
        assign(stim_config.param_values[0].second,
                transform_view(stimuli, [](cv_clamp p) { return p.second.delay; }));

        stim_config.param_values[1].first = "duration";
        assign(stim_config.param_values[1].second,
            transform_view(stimuli, [](cv_clamp p) { return p.second.duration; }));

        stim_config.param_values[2].first = "amplitude";
        assign(stim_config.param_values[2].second,
            transform_view(stimuli, [](cv_clamp p) { return p.second.amplitude; }));
    }

    return mechdata;
}

} // namespace arb
