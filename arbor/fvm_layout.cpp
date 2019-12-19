#include <set>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/morph/mcable_map.hpp>
#include <arbor/util/optional.hpp>

#include "algorithms.hpp"
#include "fvm_compartment.hpp"
#include "fvm_layout.hpp"
#include "tree.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/partition.hpp"
#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"
#include "util/transform.hpp"

namespace arb {

// Extract a branch of an mcable_map as pw_elements:

using util::pw_elements;
template <typename T>
pw_elements<T> on_branch(const mcable_map<T>& mm, msize_t bid) {
    using value_type = typename mcable_map<T>::value_type;
    pw_elements<T> pw;

    struct as_branch {
        msize_t value;
        as_branch(const value_type& x): value(x.first.branch) {}
        as_branch(const msize_t& x): value(x) {}
    };

    auto eq = std::equal_range(mm.begin(), mm.end(), bid,
            [](as_branch a, as_branch b) { return a.value<b.value; });

    for (const auto& el: util::make_range(eq)) {
        pw.push_back(el.first.prox_pos, el.first.dist_pos, el.second);
    }
    return pw;
}


using util::count_along;
using util::keys;
using util::make_span;
using util::optional;
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

    template <typename ResizableContainer, typename Index>
    auto& extend_at(ResizableContainer& c, const Index& i) {
        if (util::size(c)<=i) {
            c.resize(i+1);
        }
        return c[i];
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

    template <typename V>
    optional<V> operator|(const optional<V>& a, const optional<V>& b) {
        return a? a: b;
    }

    // For each segment given by the provided sorted sequence of segment
    // indices, call `action` for each CV intersecting the segment, starting
    // from the most proximal.
    //
    // By the ordering of CVs and segments in the discretization, with the
    // exception of the most proximal CV in a segment, each CV will be visited
    // once, and the visited CVs will be in increasing order. The most proximal
    // CV (the 'parent' CV) may be visited multiple times.
    //
    // Action is a functional that takes the following arguments:
    //
    //     size_type  cv_index   The index into the total (sorted) list of
    //                           CVs that constitute the segments.
    //
    //     index_type cv         The CV number (within the discretization).
    //
    //     value_type cv_area    The area of the CV that lies within the
    //                           current segment.
    //
    //     index_type seg_index  The index into the provided sequence of
    //                           the provided segment_indices.
    //
    //     index_type seg        The segment currently being iterated over.

    template <typename Seq, typename Action>
    void for_each_cv_in_segments(const fvm_discretization& D, const Seq& segment_indices, const Action& action) {
        using index_type = fvm_index_type;
        using size_type = fvm_size_type;

        std::unordered_map<index_type, size_type> parent_cv_indices;
        size_type cv_index = 0;

        index_type seg_index = 0;
        for (const auto& seg: segment_indices) {
            const segment_info& seg_info = D.segments[seg];

            if (seg_info.has_parent()) {
                index_type cv = seg_info.parent_cv;

                size_type i = parent_cv_indices.insert({cv, cv_index}).first->second;
                if (i==cv_index) {
                    ++cv_index;
                }

                action(i, cv, seg_info.parent_cv_area, seg_index, seg);
            }

            for (index_type cv = seg_info.proximal_cv; cv < seg_info.distal_cv; ++cv) {
                index_type i = cv_index++;
                action(i, cv, D.cv_area[cv], seg_index, seg);
            }

            index_type cv = seg_info.distal_cv;
            size_type i = cv_index++;

            action(i, seg_info.distal_cv, seg_info.distal_cv_area, seg_index, seg);
            parent_cv_indices.insert({cv, i});
            ++seg_index;
        }
    }

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

fvm_discretization fvm_discretize(const std::vector<cable_cell>& cells, const cable_cell_parameter_set& global_defaults) {

    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using size_type = fvm_size_type;

    fvm_discretization D;

    util::make_partition(D.cell_segment_bounds,
        transform_view(cells, [](const cable_cell& c) { return c.num_branches(); }));

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
    D.init_membrane_potential.assign(D.ncv, 0.);
    D.temperature_K.assign(D.ncv, 0.);
    D.diam_um.assign(D.ncv, 0.);
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

        // Electrical defaults from global defaults, possibly overridden by cell.
        auto cm_default = c.default_parameters.membrane_capacitance | global_defaults.membrane_capacitance;
        auto rL_default = c.default_parameters.axial_resistivity | global_defaults.axial_resistivity;
        auto init_vm_default = c.default_parameters.init_membrane_potential | global_defaults.init_membrane_potential;
        auto temp_default = c.default_parameters.temperature_K | global_defaults.temperature_K;

        // Compartment index range for each segment in this cell.
        seg_cv_bounds.clear();
        auto seg_cv_part = make_partition(
            seg_cv_bounds,
            transform_view(make_span(c.num_branches()), [&c](const unsigned s) {
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

        index_type soma_segment_index = D.segments.size()-1;
        D.parent_segment.push_back(soma_segment_index);

        // Other segments must all be cable segments.
        for (size_type j = 1; j<nseg; ++j) {
            const auto& seg_cv_ival = seg_cv_part[j];
            const auto ncv = seg_cv_ival.second-seg_cv_ival.first;

            segment_info seg_info;

            const auto cable = c.segment(j)->as_cable();
            if (!cable) {
                throw arbor_internal_error("fvm_layout: non-root segments of cell must be cable segments");
            }

            const auto& params = cable->parameters;
            auto cm = (params.membrane_capacitance | cm_default).value(); // [F/m²]
            auto rL = (params.axial_resistivity | rL_default).value(); // [Ω·cm]
            auto init_vm = (params.init_membrane_potential | init_vm_default).value(); // [mV]
            auto temp = (params.temperature_K | temp_default).value(); // [mV]

            bool soma_parent = c.parent(j)->as_soma() ? true : false; //segment's parent is a soma

            auto radii = cable->radii();
            auto lengths = cable->lengths();

            // If segment has soma parent, send soma information to div_compartment_integrator
            if (soma_parent) {
                radii.insert(radii.begin(), soma->radius());
                lengths.insert(lengths.begin(), soma->radius()*2);
            }

            auto divs = div_compartment_integrator(ncv, radii, lengths, soma_parent);

            seg_info.parent_cv = soma_parent ? seg_cv_ival.first : D.parent_cv[seg_cv_ival.first];
            seg_info.parent_cv_area = soma_parent ? divs(0).right.area + divs(1).left.area : divs(0).left.area;
            seg_info.soma_parent = soma_parent;

            seg_info.proximal_cv = soma_parent ? seg_cv_ival.first + 1 : seg_cv_ival.first;
            seg_info.distal_cv = seg_cv_ival.second-1;
            seg_info.distal_cv_area = divs(ncv-1).right.area;

            D.segments.push_back(seg_info);
            if (soma_parent) {
                D.parent_segment.push_back(soma_segment_index);
            }
            else {
                auto opt_index = util::binary_search_index(D.segments, seg_info.parent_cv,
                    [](const segment_info& seg_info) { return seg_info.distal_cv; });

                if (!opt_index) {
                    throw arbor_internal_error("fvm_layout: could not find parent segment");
                }
                D.parent_segment.push_back(*opt_index);
            }


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
                auto dr = div.right.radii.second*2;          // [µm]

                D.cv_area[j] += al;              // [µm²]
                D.cv_capacitance[j] += al*cm;    // [pF]
                D.init_membrane_potential[j] += al*init_vm;  // [mV·µm²]
                D.temperature_K[j] += al*temp;   // [K·µm²]

                D.cv_area[i] += ar;              // [µm²]
                D.cv_capacitance[i] += ar*cm;    // [pF]
                D.init_membrane_potential[i] += ar*init_vm;  // [mV·µm²]
                D.temperature_K[i] += ar*temp;   // [K·µm²]
                D.diam_um[i] = dr;               // [µm]
            }
        }

        auto soma_cm = (soma->parameters.membrane_capacitance | cm_default).value(); // [F/m²]
        auto soma_init_vm = (soma->parameters.init_membrane_potential | init_vm_default).value(); // [mV]
        auto soma_temp = (soma->parameters.temperature_K | temp_default).value();    // [mV]

        D.cv_area[soma_cv] = soma_area;                  // [µm²]
        D.cv_capacitance[soma_cv] = soma_area*soma_cm;   // [pF]
        D.init_membrane_potential[soma_cv] = soma_area*soma_init_vm; // [mV·µm²]
        D.temperature_K[soma_cv] = soma_area*soma_temp;  // [K·µm²]
        D.diam_um[soma_cv] = soma->radius()*2;           // [µm]
    }

    // Rescale CV init_vm and temperature values to get area-weighted means.
    for (auto i: make_span(0, D.ncv)) {
        if (D.cv_area[i]) {
            D.init_membrane_potential[i] /= D.cv_area[i]; // [mV]
            D.temperature_K[i] /= D.cv_area[i]; // [mV]
        }
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
    const cable_cell_parameter_set& gparam = gprop.default_parameters;

    fvm_mechanism_data mechdata;

    // I. Collect segment mechanism info from cells.

    // Temporary table for density mechanism info, mapping mechanism name to tuple of:
    //     1. Vector of segment indices and mechanism parameter settings where mechanism occurs.
    //     2. Set of the names of parameters that are anywhere modified.
    //     3. Pointer to mechanism metadata from catalogue.

    struct density_mech_data {
        std::vector<std::pair<size_type, const mechanism_desc*>> segments;
        string_set paramset;
        std::shared_ptr<mechanism_info> info = nullptr;
    };
    std::unordered_map<std::string, density_mech_data> density_mech_table;

    // Temporary table for revpot mechanism info, mapping mechanism name to tuple of:
    //     1. Sorted map from cell index to mechanism parameter settings.
    //     2. Set of the names of parameters that are anywhere modified.
    //     3. Pointer to mechanism metadat from catalogue.

    struct revpot_mech_data {
        std::map<size_type, const mechanism_desc*> cells;
        string_set paramset;
        std::shared_ptr<mechanism_info> info = nullptr;
    };
    std::unordered_map<std::string, revpot_mech_data> revpot_mech_table;

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
        std::shared_ptr<mechanism_info> info = nullptr;
    };
    std::unordered_map<std::string, point_mech_data> point_mech_table;

    // Built-in stimulus mechanism data is dealt with especially below.
    // Record for each stimulus the CV and clamp data.

    std::vector<std::pair<size_type, i_clamp>> stimuli;

    // Temporary table for presence of ion channels, mapping ion name to a _sorted_
    // collection of per-segment ion data, viz. a map from segment index to
    // initial internal and external concentrations and reversal potentials, plus
    // information about whether there is a mechanism that requires this ion
    // on this segment, and if that ion writes to internal or external concentration.

    struct ion_segment_data {
        cable_cell_ion_data ion_data;
        bool mech_requires = false;
        bool mech_writes_iconc = false;
        bool mech_writes_econc = false;
    };

    std::unordered_map<std::string, std::map<index_type, ion_segment_data>> ion_segments;

    // Temporary table for presence of mechanisms that read the reversal potential
    // of an ion channel, mapping ion name and cell index to a _sorted_
    // collection of segment indices.

    std::unordered_map<std::string, std::unordered_map<size_type, std::set<size_type>>> ion_revpot_segments;

    auto verify_ion_usage =
        [&gprop](const std::string& mech_name, const mechanism_info* info)
    {
        const auto& global_ions = gprop.ion_species;

        for (const auto& ion: info->ions) {
            const auto& ion_name = ion.first;
            const auto& ion_dep = ion.second;

            if (!global_ions.count(ion_name)) {
                throw cable_cell_error(
                    "mechanism "+mech_name+" uses ion "+ion_name+ " which is missing in global properties");
            }

            if (ion_dep.verify_ion_charge) {
                if (ion_dep.expected_ion_charge!=global_ions.at(ion_name)) {
                    throw cable_cell_error(
                        "mechanism "+mech_name+" uses ion "+ion_name+ " expecting a different valence");
                }
            }
        }
    };

    auto update_paramset_and_validate =
        [&catalogue,&verify_ion_usage]
        (const mechanism_desc& desc, std::shared_ptr<mechanism_info>& info, string_set& paramset)
    {
        auto& name = desc.name();
        if (!info) {
            info.reset(new mechanism_info(catalogue[name]));
            verify_ion_usage(name, info.get());
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

    auto cell_segment_part = D.cell_segment_part();
    size_type target_id = 0;

    for (auto cell_idx: make_span(0, D.ncell)) {
        auto& cell = cells[cell_idx];
        auto seg_range = cell_segment_part[cell_idx];

        auto add_ion_segment =
            [&gparam, &cell, &ion_segments]
            (const std::string& ion_name, size_type segment_idx, const cable_cell_parameter_set& seg_param, const ion_dependency* iondep = nullptr)
        {
            const auto& global_ion_data = gparam.ion_data;
            const auto& cell_ion_data = cell.default_parameters.ion_data;
            const auto& seg_ion_data = seg_param.ion_data;

            auto& ion_entry = ion_segments[ion_name];

            // New entry?
            util::optional<cable_cell_ion_data> opt_ion_data;
            if (!ion_entry.count(segment_idx)) {
                opt_ion_data = value_by_key(seg_ion_data, ion_name);
                if (!opt_ion_data) {
                    opt_ion_data = value_by_key(cell_ion_data, ion_name);
                }
                if (!opt_ion_data) {
                    opt_ion_data = value_by_key(global_ion_data, ion_name);
                }
                if (!opt_ion_data) {
                    throw arbor_internal_error("missing entry for ion "+ion_name+" in cable_cell global defaults");
                }
            }

            auto& ion_entry_seg = ion_entry[segment_idx];
            if (opt_ion_data) {
                ion_entry_seg.ion_data = *opt_ion_data;
            }
            if (iondep) {
                ion_entry_seg.mech_requires = true;
                ion_entry_seg.mech_writes_iconc |= iondep->write_concentration_int;
                ion_entry_seg.mech_writes_econc |= iondep->write_concentration_ext;
            }
        };

        for (auto segment_idx: make_span(seg_range)) {
            const segment_ptr& seg = cell.segments()[segment_idx-seg_range.first];

            for (const mechanism_desc& desc: seg->mechanisms()) {
                const auto& name = desc.name();

                density_mech_data& entry = density_mech_table[name];
                update_paramset_and_validate(desc, entry.info, entry.paramset);
                entry.segments.emplace_back(segment_idx, &desc);

                for (const auto& ion_entry: entry.info->ions) {
                    const std::string& ion_name = ion_entry.first;
                    const ion_dependency& iondep = ion_entry.second;

                    add_ion_segment(ion_name, segment_idx, seg->parameters, &iondep);

                    if (ion_entry.second.read_reversal_potential) {
                        ion_revpot_segments[ion_name][cell_idx].insert(segment_idx);
                    }
                }
            }
        }

        for (const auto& cellsyn: cell.synapses()) {
            const mechanism_desc& desc = cellsyn.mechanism;
            size_type cv = D.branch_location_cv(cell_idx, cellsyn.location);
            const auto& name = desc.name();

            point_mech_data& entry = point_mech_table[name];
            update_paramset_and_validate(desc, entry.info, entry.paramset);
            entry.points.push_back({cv, target_id++, &desc});

            const segment_ptr& seg = cell.segments()[cellsyn.location.branch];
            size_type segment_idx = D.cell_segment_bounds[cell_idx]+cellsyn.location.branch;

            for (const auto& ion_entry: entry.info->ions) {
                const std::string& ion_name = ion_entry.first;
                const ion_dependency& iondep = ion_entry.second;

                add_ion_segment(ion_name, segment_idx, seg->parameters, &iondep);

                if (ion_entry.second.read_reversal_potential) {
                    ion_revpot_segments[ion_name][cell_idx].insert(segment_idx);
                }
            }
        }

        for (const auto& stimulus: cell.stimuli()) {
            size_type cv = D.branch_location_cv(cell_idx, stimulus.location);
            stimuli.push_back({cv, stimulus.clamp});
        }

        // Add segments to ion_segments map which intersect with existing segments, so
        // that each CV with an ion value is 100% covered.

        for (auto& e: ion_segments) {
            const auto& name = e.first;
            const auto& ion_entry = e.second;

            for (auto segment_idx: make_span(seg_range)) {
                index_type parent_segment_idx = D.parent_segment[segment_idx];
                const segment_ptr& parent_seg = cell.segments()[parent_segment_idx-seg_range.first];

                if (ion_entry.count(segment_idx)) {
                    add_ion_segment(name, parent_segment_idx, parent_seg->parameters);
                }
            }

            for (auto segment_idx: make_span(seg_range)) {
                const segment_ptr& seg = cell.segments()[segment_idx-seg_range.first];
                index_type parent_segment_idx = D.parent_segment[segment_idx];

                if (ion_entry.count(parent_segment_idx)) {
                    add_ion_segment(name, segment_idx, seg->parameters);
                }
            }
        }


        // Maintain a map of reversal potential mechanism to written ions for this
        // cell, to ensure consistency with assignments from the cell and global
        // parameters.

        std::unordered_multimap<std::string, std::string> revpot_to_ion;
        std::unordered_map<std::string, std::string> ion_to_revpot;

        auto add_revpot = [&](const std::string& ion, const mechanism_desc& desc) {
            const auto& name = desc.name();

            revpot_mech_data& entry = revpot_mech_table[name];
            update_paramset_and_validate(desc, entry.info, entry.paramset);
            ion_to_revpot[ion] = desc.name();
            for (auto dep: entry.info->ions) {
                if (dep.second.write_reversal_potential) {
                    revpot_to_ion.insert({desc.name(), dep.first});
                }
            }
            entry.cells.insert({cell_idx, &desc});
        };

        const auto& cellrevpot = cell.default_parameters.reversal_potential_method;
        for (const auto& revpot: cellrevpot) {
            add_revpot(revpot.first, revpot.second);
        }

        const auto& globalrevpot =  gparam.reversal_potential_method;
        for (const auto& revpot: globalrevpot) {
            if (!cellrevpot.count(revpot.first)) {
                add_revpot(revpot.first, revpot.second);
            }
        }

        // Ensure that if a revpot mechanism writes to multiple ions, that
        // that mechanism is associated with all of them.
        for (auto& entry: revpot_to_ion) {
            auto declared_revpot = value_by_key(ion_to_revpot, entry.second);
            if (!declared_revpot || declared_revpot.value()!=entry.first) {
                throw cable_cell_error("mechanism "+entry.first+" writes reversal potential for "
                    +entry.second+", which is not configured to use "+entry.first);
            }
        }
    }


    // II. Build ion and mechanism configs.

    // Shared temporary lookup info across mechanism instances, set by build_param_data.
    string_index_map param_index;           // maps parameter name to parameter index
    std::vector<std::string> param_name;    // indexed by parameter index
    std::vector<value_type> param_default;  // indexed by parameter index

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

    for (const auto& ionseg: ion_segments) {
        auto& seg_ion_map = ion_segments[ionseg.first];
        fvm_ion_config& ion_config = mechdata.ions[ionseg.first];
        auto& seg_ion_data = ionseg.second;

        for_each_cv_in_segments(D, keys(seg_ion_data),
            [&ion_config, &seg_ion_map](size_type cv_index, index_type cv, value_type area, index_type seg_index, index_type seg) {
                if (seg_ion_map[seg].mech_requires) {
                    if (!util::binary_search_index(ion_config.cv, cv)) {
                        ion_config.cv.push_back(cv);
                    }
                }
            });

        ion_config.init_iconc.resize(ion_config.cv.size());
        ion_config.init_econc.resize(ion_config.cv.size());
        ion_config.reset_iconc.resize(ion_config.cv.size());
        ion_config.reset_econc.resize(ion_config.cv.size());
        ion_config.init_revpot.resize(ion_config.cv.size());

        for_each_cv_in_segments(D, keys(seg_ion_data),
            [&ion_config, &seg_ion_map, &D](size_type cv_index, index_type cv, value_type area, index_type seg_index, index_type seg) {
                auto opt_i = util::binary_search_index(ion_config.cv, cv);
                if (!opt_i) return;

                std::size_t i = *opt_i;
                auto& seg_ion_entry = seg_ion_map[seg];

                value_type weight = area/D.cv_area[cv];
                ion_config.reset_iconc[i] += weight*seg_ion_entry.ion_data.init_int_concentration;
                ion_config.reset_econc[i] += weight*seg_ion_entry.ion_data.init_ext_concentration;

                if (!seg_ion_entry.mech_writes_iconc) {
                    ion_config.init_iconc[i] += weight*seg_ion_entry.ion_data.init_int_concentration;
                }

                if (!seg_ion_entry.mech_writes_econc) {
                    ion_config.init_econc[i] += weight*seg_ion_entry.ion_data.init_ext_concentration;
                }

                // Reversal potentials are not area weighted, and are overridden at the
                // per-cell level by any supplied revpot mechanisms.
                ion_config.init_revpot[i] = seg_ion_entry.ion_data.init_reversal_potential;
            });
    }

    // IIb. Reversal potential mechanism CVs and parameters.

    for (const auto& entry: revpot_mech_table) {
        const std::string& name = entry.first;
        const mechanism_info& info = *entry.second.info;

        fvm_mechanism_config& config = mechdata.mechanisms[name];
        config.kind = mechanismKind::revpot;

        auto nparam = build_param_data(entry.second.paramset, &info);
        config.param_values.resize(nparam);
        for (auto pidx: make_span(nparam)) {
            config.param_values[pidx].first = param_name[pidx];
        }

        for (auto& cell_entry: entry.second.cells) {
            auto cell_idx = cell_entry.first;
            std::vector<index_type> segment_indices;

            for (auto& mech_ion_dep_entry: info.ions) {
                auto& ion_name = mech_ion_dep_entry.first;
                auto& ion_dep = mech_ion_dep_entry.second;

                if (!ion_dep.write_reversal_potential) continue;

                const auto& segments = value_by_key(ion_revpot_segments[ion_name], cell_idx);
                if (!segments) continue;

                for (auto seg_idx: segments.value()) {
                    segment_indices.push_back(seg_idx);
                }
            }

            const mechanism_desc& desc = *cell_entry.second;
            std::vector<value_type> pval = param_default;

            for (auto pidx: make_span(nparam)) {
                if (auto opt_v = value_by_key(desc.values(), param_name[pidx])) {
                    pval[pidx] = opt_v.value();
                }
            }

            size_type config_offset = config.cv.size();
            for_each_cv_in_segments(D, segment_indices,
                [&](size_type cv_index, index_type cv, auto, auto, auto) {
                    extend_at(config.cv, config_offset+cv_index) = cv;

                    for (auto pidx: make_span(nparam)) {
                        extend_at(config.param_values[pidx].second, config_offset+cv_index) = pval[pidx];
                    }
                });
        }
    }

    // Remove any reversal potential mechanisms that ultimately have no extent.
    for (auto i = mechdata.mechanisms.begin(); i!=mechdata.mechanisms.end(); ) {
        i = i->second.cv.empty()? mechdata.mechanisms.erase(i): std::next(i);
    }

    // IIc. Density mechanism CVs, parameters and ionic default concentration contributions.

    // Ameliorate area sum rounding errors by clamping normalized area contributions to [0, 1]
    // and rounding values within an epsilon of 0 or 1 to that value.
    auto trim = [](value_type& v) {
        constexpr value_type eps = std::numeric_limits<value_type>::epsilon()*4;
        v = v<eps? 0: v+eps>1? 1: v;
    };

    for (const auto& entry: density_mech_table) {
        const std::string& name = entry.first;
        fvm_mechanism_config& config = mechdata.mechanisms[name];
        config.kind = mechanismKind::density;

        auto nparam = build_param_data(entry.second.paramset, entry.second.info.get());

        // In order to properly account for partially overriden parameters in CVs
        // that are shared between segments, we need to track not only the area-weighted
        // sum of parameter values, but also the total area for each CV for each parameter
        // that has been overriden — the remaining area demands a contribution from the
        // parameter default value.

        std::vector<std::vector<value_type>> param_value(nparam);
        std::vector<std::vector<value_type>> param_area_contrib(nparam);

        for_each_cv_in_segments(D, keys(entry.second.segments),
            [&](size_type cv_index, index_type cv, value_type area, index_type seg_index, index_type seg)
            {
                const mechanism_desc& mech_desc = *entry.second.segments[seg_index].second;

                extend_at(config.cv, cv_index) = cv;

                for (auto& kv: mech_desc.values()) {
                    int pidx = param_index.at(kv.first);
                    value_type v = kv.second;

                    extend_at(param_area_contrib[pidx], cv_index) += area;
                    extend_at(param_value[pidx], cv_index) += area*v;
                }

                extend_at(config.norm_area, cv_index) += area;
            });

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

    // II.3 Point mechanism CVs, targets, parameters and stimuli.

    for (const auto& entry: point_mech_table) {
        const std::string& name = entry.first;
        const auto& points = entry.second.points;

        auto nparam = build_param_data(entry.second.paramset, entry.second.info.get());
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
        if (catalogue[name].linear && gprop.coalesce_synapses) {
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
