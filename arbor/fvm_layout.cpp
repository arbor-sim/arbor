#include <algorithm>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/math.hpp>
#include <arbor/morph/mcable_map.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/util/optional.hpp>

#include "fvm_layout.hpp"
#include "threading/threading.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/partition.hpp"
#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"
#include "util/transform.hpp"
#include "util/unique.hpp"

namespace arb {

using util::append;
using util::assign;
using util::assign_by;
using util::count_along;
using util::pw_elements;
using util::pw_element;
using util::sort;
using util::sort_by;
using util::stable_sort_by;
using util::value_by_key;

namespace {
struct get_value {
    template <typename X>
    double operator()(const X& x) const { return x.value; }

    double operator()(double x) const { return x; }
};

template <typename V>
util::optional<V> operator|(const util::optional<V>& a, const util::optional<V>& b) {
    return a? a: b;
}

// Given sorted vectors a, b, return sorted vector with unique elements v
// such that v is present in a or b.
template <typename V>
std::vector<V> unique_union(const std::vector<V>& a, const std::vector<V>& b) {
    std::vector<V> u;

    auto ai = a.begin();
    auto ae = a.end();
    auto bi = b.begin();
    auto be = b.end();

    while (ai!=ae && bi!=be) {
        const V& elem = *ai<*bi? *ai++: *bi++;
        if (u.empty() || u.back()!=elem) {
            u.push_back(elem);
        }
    }

    while (ai!=ae) {
        const V& elem = *ai++;
        if (u.empty() || u.back()!=elem) {
            u.push_back(elem);
        }
    }

    while (bi!=be) {
        const V& elem = *bi++;
        if (u.empty() || u.back()!=elem) {
            u.push_back(elem);
        }
    }
    return u;
}

} // anonymous namespace

// Convert mcable_map values to a piecewise function over an mcable.
// The projection gives the map from the values in the mcable_map to the values in the piecewise function.
template <typename T, typename U, typename Proj = get_value>
pw_elements<U> pw_over_cable(const mcable_map<T>& mm, mcable cable, U dflt_value, Proj projection = Proj{}) {
    using value_type = typename mcable_map<T>::value_type;
    msize_t bid = cable.branch;

    struct as_branch {
        msize_t value;
        as_branch(const value_type& x): value(x.first.branch) {}
        as_branch(const msize_t& x): value(x) {}
    };

    auto map_on_branch = util::make_range(
            std::equal_range(mm.begin(), mm.end(), bid,
                [](as_branch a, as_branch b) { return a.value<b.value; }));

    if (map_on_branch.empty()) {
        return pw_elements<U>({cable.prox_pos, cable.dist_pos}, {dflt_value});
    }

    pw_elements<U> pw;
    for (const auto& el: map_on_branch) {
        double pw_right = pw.empty()? 0: pw.bounds().second;
        if (el.first.prox_pos>pw_right) {
            pw.push_back(pw_right, el.first.prox_pos, dflt_value);
        }
        pw.push_back(el.first.prox_pos, el.first.dist_pos, projection(el.second));
    }

    double pw_right = pw.empty()? 0: pw.bounds().second;
    if (pw_right<1.) {
        pw.push_back(pw_right, 1., dflt_value);
    }

    if (cable.prox_pos!=0 || cable.dist_pos!=1) {
        pw = zip(pw, pw_elements<>({cable.prox_pos, cable.dist_pos}));
    }
    return pw;
}

// Construct cv_geometry for cell from locset describing CV boundary points.
cv_geometry cv_geometry_from_ends(const cable_cell& cell, const locset& lset) {
    auto pop = [](auto& vec) { auto h = vec.back(); return vec.pop_back(), h; };

    cv_geometry geom;
    const auto& mp = cell.provider();
    const auto& m = mp.morphology();

    if (m.empty()) {
        geom.cell_cv_divs = {0, 0};
        return geom;
    }

    mlocation_list locs = thingify(lset, mp);

    // Filter out root, terminal locations and repeated locations so as to
    // avoid trivial CVs outside of fork points. (This is not necessary for
    // correctness, but is for the convenience of specification by lset.)

    auto neither_root_nor_terminal = [&m](mlocation x) {
        return !(x.pos==0 && x.branch==(m.branch_children(mnpos).size()>1u? mnpos: 0)) // root?
            && !(x.pos==1 && m.branch_children(x.branch).empty()); // terminal?
    };
    locs.erase(std::partition(locs.begin(), locs.end(), neither_root_nor_terminal), locs.end());
    util::sort(locs);
    util::unique_in_place(locs);

    // Collect cables constituting each CV, maintaining a stack of CV
    // proximal 'head' points, and recursing down branches in the morphology
    // within each CV.

    constexpr fvm_index_type no_parent = -1;
    std::vector<std::pair<mlocation, fvm_index_type>> next_cv_head; // head loc, parent cv index
    next_cv_head.emplace_back(mlocation{mnpos, 0}, no_parent);

    mcable_list cables;
    std::vector<msize_t> branches;
    geom.cv_cables_divs.push_back(0);
    fvm_index_type cv_index = 0;

    while (!next_cv_head.empty()) {
        auto next = pop(next_cv_head);
        mlocation h = next.first;

        cables.clear();
        branches.clear();
        branches.push_back(h.branch);
        geom.cv_parent.push_back(next.second);

        while (!branches.empty()) {
            msize_t b = pop(branches);

            // Find most proximal point in locs on this branch, strictly more distal than h.
            auto it = locs.end();
            if (b!=mnpos && b==h.branch) {
                it = std::upper_bound(locs.begin(), locs.end(), h);
            }
            else if (b!=mnpos) {
                it = std::lower_bound(locs.begin(), locs.end(), mlocation{b, 0});
            }

            // If found, use as an end point, and stop descent.
            // Otherwise, recurse over child branches.
            if (it!=locs.end() && it->branch==b) {
                cables.push_back({b, b==h.branch? h.pos: 0, it->pos});
                next_cv_head.emplace_back(*it, cv_index);
            }
            else {
                if (b!=mnpos) {
                    cables.push_back({b, b==h.branch? h.pos: 0, 1});
                }
                for (auto& c: m.branch_children(b)) {
                    branches.push_back(c);
                }
            }
        }

        sort(cables);
        append(geom.cv_cables, std::move(cables));
        geom.cv_cables_divs.push_back(geom.cv_cables.size());
        ++cv_index;
    }

    auto n_cv = cv_index;
    arb_assert(n_cv>0);
    arb_assert(geom.cv_parent.front()==-1);
    arb_assert(util::all_of(util::subrange_view(geom.cv_parent, 1, n_cv),
            [](auto v) { return v!=no_parent; }));

    // Construct CV children mapping by sorting CV indices by parent.
    assign(geom.cv_children, util::make_span(1, n_cv));
    stable_sort_by(geom.cv_children, [&geom](auto cv) { return geom.cv_parent[cv]; });

    geom.cv_children_divs.reserve(n_cv+1);
    geom.cv_children_divs.push_back(0);

    auto b = geom.cv_children.begin();
    auto e = geom.cv_children.end();
    auto from = b;

    for (fvm_index_type cv = 0; cv<n_cv; ++cv) {
        from = std::partition_point(from, e,
            [cv, &geom](auto i) { return geom.cv_parent[i]<=cv; });
        geom.cv_children_divs.push_back(from-b);
    }

    // Fill cv/cell mapping for single cell (index 0).
    geom.cv_to_cell.assign(cv_index, 0);
    geom.cell_cv_divs = {0, (fvm_index_type)cv_index};

    // Build location query map.
    geom.branch_cv_map.resize(1);
    std::vector<pw_elements<fvm_size_type>>& bmap = geom.branch_cv_map.back();

    for (auto cv: util::make_span(n_cv)) {
        for (auto cable: geom.cables(cv)) {
            if (cable.branch>=bmap.size()) {
                bmap.resize(cable.branch+1);
            }

            // Ordering of CV ensures CV cables on any given branch are found sequentially.
            bmap[cable.branch].push_back(cable.prox_pos, cable.dist_pos, cv);
        }
    }

    return geom;
}

namespace impl {
    using std::begin;
    using std::end;
    using std::next;

    template <typename Seq>
    auto tail(Seq& seq) { return util::make_range(next(begin(seq)), end(seq)); };

    template <typename Container, typename Offset, typename Seq>
    void append_offset(Container& ctr, Offset offset, const Seq& rhs) {
        for (const auto& x: rhs) {
            // Preserve -1 'npos' values.
            ctr.push_back(x+1==0? x: offset+x);
        }
    }
}

// Merge CV geometry lists in-place.
cv_geometry& append(cv_geometry& geom, const cv_geometry& right) {
    using impl::tail;
    using impl::append_offset;

    if (!right.n_cell()) {
        return geom;
    }

    if (!geom.n_cell()) {
        geom = right;
        return geom;
    }

    auto append_divs = [](auto& left, const auto& right) {
        if (left.empty()) {
            left = right;
        }
        else if (!right.empty()) {
            append_offset(left, left.back(), tail(right));
        }
    };

    auto geom_n_cv = geom.size();
    auto geom_n_cell = geom.n_cell();

    append(geom.cv_cables, right.cv_cables);
    append_divs(geom.cv_cables_divs, right.cv_cables_divs);

    append_offset(geom.cv_parent, geom_n_cv, right.cv_parent);
    append_offset(geom.cv_children, geom_n_cv, right.cv_children);
    append_divs(geom.cv_children_divs, right.cv_children_divs);

    append_offset(geom.cv_to_cell, geom_n_cell, right.cv_to_cell);
    append_divs(geom.cell_cv_divs, right.cell_cv_divs);

    append(geom.branch_cv_map, right.branch_cv_map);
    return geom;
}

// Combine two fvm_cv_geometry groups in-place.
fvm_cv_discretization& append(fvm_cv_discretization& dczn, const fvm_cv_discretization& right) {
    append(dczn.geometry, right.geometry);

    append(dczn.face_conductance, right.face_conductance);
    append(dczn.cv_area, right.cv_area);
    append(dczn.cv_capacitance, right.cv_capacitance);
    append(dczn.init_membrane_potential, right.init_membrane_potential);
    append(dczn.temperature_K, right.temperature_K);
    append(dczn.diam_um, right.diam_um);

    append(dczn.axial_resistivity, right.axial_resistivity);

    return dczn;
}

fvm_cv_discretization fvm_cv_discretize(const cable_cell& cell, const cable_cell_parameter_set& global_dflt) {
    const auto& dflt = cell.default_parameters;
    fvm_cv_discretization D;

    D.geometry = cv_geometry_from_ends(cell,
        dflt.discretization? dflt.discretization->cv_boundary_points(cell):
        global_dflt.discretization? global_dflt.discretization->cv_boundary_points(cell):
        default_cv_policy().cv_boundary_points(cell));

    if (D.geometry.empty()) return D;

    auto n_cv = D.geometry.size();
    D.face_conductance.resize(n_cv);
    D.cv_area.resize(n_cv);
    D.cv_capacitance.resize(n_cv);
    D.init_membrane_potential.resize(n_cv);
    D.temperature_K.resize(n_cv);
    D.diam_um.resize(n_cv);

    double dflt_resistivity = *(dflt.axial_resistivity | global_dflt.axial_resistivity);
    double dflt_capacitance = *(dflt.membrane_capacitance | global_dflt.membrane_capacitance);
    double dflt_potential =   *(dflt.init_membrane_potential | global_dflt.init_membrane_potential);
    double dflt_temperature = *(dflt.temperature_K | global_dflt.temperature_K);

    D.axial_resistivity.resize(1);
    msize_t n_branch = D.geometry.n_branch(0);
    D.axial_resistivity[0].reserve(n_branch);
    for (msize_t i = 0; i<n_branch; ++i) {
        D.axial_resistivity[0].push_back(pw_over_cable(cell.region_assignments().get<axial_resistivity>(),
                    mcable{i, 0., 1.}, dflt_resistivity));
    }

    const auto& embedding = cell.embedding();
    for (auto i: count_along(D.geometry.cv_parent)) {
        auto cv_cables = D.geometry.cables(i);

        // Computing face_conductance:
        //
        // Flux between adjacent CVs is computed as if there were no membrane currents, and with the CV voltage
        // values taken to be exact at a reference point in each CV:
        //     * If the CV is unbranched, the reference point is taken to be the CV midpoint.
        //     * If the CV is branched, the reference point is taken to be closest branch point to
        //       the interface between the two CVs.

        D.face_conductance[i] = 0;

        fvm_index_type p = D.geometry.cv_parent[i];
        if (p!=-1) {
            auto parent_cables = D.geometry.cables(p);

            msize_t bid = cv_cables.front().branch;
            double parent_refpt = 0;
            double cv_refpt = 1;

            if (cv_cables.size()==1) {
                mcable cv_cable = cv_cables.front();
                cv_refpt = 0.5*(cv_cable.prox_pos+cv_cable.dist_pos);
            }
            if (parent_cables.size()==1) {
                mcable parent_cable = parent_cables.front();
                // A trivial parent CV with a zero-length cable might not
                // be on the same branch.
                if (parent_cable.branch==bid) {
                    parent_refpt = 0.5*(parent_cable.prox_pos+parent_cable.dist_pos);
                }
            }

            mcable span{bid, parent_refpt, cv_refpt};
            double resistance = embedding.integrate_ixa(span, D.axial_resistivity[0].at(bid));
            D.face_conductance[i] = 100/resistance; // 100 scales to µS.
        }

        D.cv_area[i] = 0;
        D.cv_capacitance[i] = 0;
        D.init_membrane_potential[i] = 0;
        D.diam_um[i] = 0;
        double cv_length = 0;

        for (mcable c: cv_cables) {
            D.cv_area[i] += embedding.integrate_area(c);

            D.cv_capacitance[i] += embedding.integrate_area(c.branch,
                pw_over_cable(cell.region_assignments().get<membrane_capacitance>(), c, dflt_capacitance));

            D.init_membrane_potential[i] += embedding.integrate_area(c.branch,
                pw_over_cable(cell.region_assignments().get<init_membrane_potential>(), c, dflt_potential));

            D.temperature_K[i] += embedding.integrate_area(c.branch,
                pw_over_cable(cell.region_assignments().get<temperature_K>(), c, dflt_temperature));

            cv_length += embedding.integrate_length(c);
        }

        if (D.cv_area[i]>0) {
            D.init_membrane_potential[i] /= D.cv_area[i];
            D.temperature_K[i] /= D.cv_area[i];
        }
        if (cv_length>0) {
            D.diam_um[i] = D.cv_area[i]/(cv_length*math::pi<double>);
        }
    }

    return D;
}

fvm_cv_discretization fvm_cv_discretize(const std::vector<cable_cell>& cells,
    const cable_cell_parameter_set& global_defaults,
    const arb::execution_context& ctx)
{
    std::vector<fvm_cv_discretization> cell_disc(cells.size());
    threading::parallel_for::apply(0, cells.size(), ctx.thread_pool.get(),
          [&] (int i) { cell_disc[i]=fvm_cv_discretize(cells[i], global_defaults);});

    fvm_cv_discretization combined;
    for (auto cell_idx: count_along(cells)) {
        append(combined, cell_disc[cell_idx]);
    }
    return combined;
}

// CVs are absolute (taken from combined discretization) so do not need to be shifted.
// Only target numbers need to be shifted.
fvm_mechanism_data& append(fvm_mechanism_data& left, const fvm_mechanism_data& right) {
    using impl::append_offset;

    fvm_size_type target_offset = left.n_target;

    for (const auto& kv: right.ions) {
        fvm_ion_config& L = left.ions[kv.first];
        const fvm_ion_config& R = kv.second;

        append(L.cv, R.cv);
        append(L.init_iconc, R.init_iconc);
        append(L.init_econc, R.init_econc);
        append(L.reset_iconc, R.reset_iconc);
        append(L.reset_econc, R.reset_econc);
        append(L.init_revpot, R.init_revpot);
    }

    for (const auto& kv: right.mechanisms) {
        if (!left.mechanisms.count(kv.first)) {
            fvm_mechanism_config& L = left.mechanisms[kv.first];

            L = kv.second;
            for (auto& t: L.target) t += target_offset;
        }
        else {
            fvm_mechanism_config& L = left.mechanisms[kv.first];
            const fvm_mechanism_config& R = kv.second;

            L.kind = R.kind;
            append(L.cv, R.cv);
            append(L.multiplicity, R.multiplicity);
            append(L.norm_area, R.norm_area);
            append_offset(L.target, target_offset, R.target);

            arb_assert(util::is_sorted_by(L.param_values, util::first));
            arb_assert(util::is_sorted_by(R.param_values, util::first));
            arb_assert(L.param_values.size()==R.param_values.size());

            for (auto j: count_along(R.param_values)) {
                arb_assert(L.param_values[j].first==R.param_values[j].first);
                append(L.param_values[j].second, R.param_values[j].second);
            }
        }
    }
    left.n_target += right.n_target;

    return left;
}

fvm_mechanism_data fvm_build_mechanism_data(const cable_cell_global_properties& gprop,
    const cable_cell& cell, const fvm_cv_discretization& D, fvm_size_type cell_idx);

fvm_mechanism_data fvm_build_mechanism_data(const cable_cell_global_properties& gprop,
    const std::vector<cable_cell>& cells, const fvm_cv_discretization& D, const execution_context& ctx)
{
    std::vector<fvm_mechanism_data> cell_mech(cells.size());
    threading::parallel_for::apply(0, cells.size(), ctx.thread_pool.get(),
          [&] (int i) { cell_mech[i]=fvm_build_mechanism_data(gprop, cells[i], D, i);});

    fvm_mechanism_data combined;
    for (auto cell_idx: count_along(cells)) {
        append(combined, cell_mech[cell_idx]);
    }
    return combined;
}

fvm_mechanism_data fvm_build_mechanism_data(const cable_cell_global_properties& gprop,
    const cable_cell& cell, const fvm_cv_discretization& D, fvm_size_type cell_idx)
{
    using size_type = fvm_size_type;
    using index_type = fvm_index_type;
    using value_type = fvm_value_type;

    const mechanism_catalogue& catalogue = *gprop.catalogue;
    const auto& embedding = cell.embedding();

    const auto& global_dflt = gprop.default_parameters;
    const auto& dflt = cell.default_parameters;

    fvm_mechanism_data M;

    // Verify mechanism ion usage, parameter values.
    auto verify_mechanism = [&gprop](const mechanism_info& info, const mechanism_desc& desc) {
        const auto& global_ions = gprop.ion_species;

        for (const auto& pv: desc.values()) {
            if (!info.parameters.count(pv.first)) {
                throw no_such_parameter(desc.name(), pv.first);
            }
            if (!info.parameters.at(pv.first).valid(pv.second)) {
                throw invalid_parameter_value(desc.name(), pv.first, pv.second);
            }
        }

        for (const auto& ion: info.ions) {
            const auto& ion_name = ion.first;
            const auto& ion_dep = ion.second;

            if (!global_ions.count(ion_name)) {
                throw cable_cell_error(
                    "mechanism "+desc.name()+" uses ion "+ion_name+ " which is missing in global properties");
            }

            if (ion_dep.verify_ion_charge) {
                if (ion_dep.expected_ion_charge!=global_ions.at(ion_name)) {
                    throw cable_cell_error(
                        "mechanism "+desc.name()+" uses ion "+ion_name+ " expecting a different valence");
                }
            }

            if (ion_dep.write_reversal_potential && (ion_dep.write_concentration_int || ion_dep.write_concentration_ext)) {
                throw cable_cell_error("mechanism "+desc.name()+" writes both reversal potential and concentration");
            }
        }
    };

    // Track ion usage of mechanisms so that ions are only instantiated where required.
    std::unordered_map<std::string, std::vector<index_type>> ion_support;
    auto update_ion_support = [&ion_support](const mechanism_info& info, const std::vector<index_type>& cvs) {
        arb_assert(util::is_sorted(cvs));

        for (const auto& ion: util::keys(info.ions)) {
            auto& support = ion_support[ion];
            support = unique_union(support, cvs);
        }
    };

    std::unordered_map<std::string, mcable_map<double>> init_iconc_mask;
    std::unordered_map<std::string, mcable_map<double>> init_econc_mask;

    // Density mechanisms:

    for (const auto& entry: cell.region_assignments().get<mechanism_desc>()) {
        const std::string& name = entry.first;
        mechanism_info info = catalogue[name];

        std::vector<double> param_dflt;
        fvm_mechanism_config config;
        config.kind = mechanismKind::density;

        std::vector<std::string> param_names;
        assign(param_names, util::keys(info.parameters));
        sort(param_names);

        for (auto& p: param_names) {
            config.param_values.emplace_back(p, std::vector<value_type>{});
            param_dflt.push_back(info.parameters.at(p).default_value);
        }

        mcable_map<double> support;
        std::vector<mcable_map<double>> param_maps;

        {
            std::unordered_map<std::string, mcable_map<double>> keyed_param_maps;
            for (auto& on_cable: entry.second) {
                verify_mechanism(info, on_cable.second);
                mcable cable = on_cable.first;

                support.insert(cable, 1.);
                for (auto param_assign: on_cable.second.values()) {
                    keyed_param_maps[param_assign.first].insert(cable, param_assign.second);
                }
            }
            for (auto& p: param_names) {
                param_maps.push_back(std::move(keyed_param_maps[p]));
            }
        }

        std::vector<double> param_on_cv(config.param_values.size());

        for (auto cv: D.geometry.cell_cvs(cell_idx)) {
            double area = 0;
            util::fill(param_on_cv, 0.);

            for (mcable c: D.geometry.cables(cv)) {
                double area_on_cable = embedding.integrate_area(c.branch, pw_over_cable(support, c, 0.));
                if (!area_on_cable) continue;

                area += area_on_cable;
                for (auto i: count_along(param_on_cv)) {
                    param_on_cv[i] += embedding.integrate_area(c.branch, pw_over_cable(param_maps[i], c, param_dflt[i]));
                }
            }

            if (area>0) {
                double oo_cv_area = 1./D.cv_area[cv];
                config.cv.push_back(cv);
                config.norm_area.push_back(area*oo_cv_area);
                for (auto i: count_along(param_on_cv)) {
                    config.param_values[i].second.push_back(param_on_cv[i]*oo_cv_area);
                }
            }
        }

        for (const auto& iondep: info.ions) {
            if (iondep.second.write_concentration_int) {
                for (auto c: support) {
                    bool ok = init_iconc_mask[iondep.first].insert(c.first, 0.);
                    if (!ok) {
                        throw cable_cell_error("overlapping ion concentration writing mechanism "+name);
                    }
                }
            }
            if (iondep.second.write_concentration_ext) {
                for (auto c: support) {
                    bool ok = init_econc_mask[iondep.first].insert(c.first, 0.);
                    if (!ok) {
                        throw cable_cell_error("overlapping ion concentration writing mechanism "+name);
                    }
                }
            }
        }

        update_ion_support(info, config.cv);
        M.mechanisms[name] = std::move(config);
    }

    // Synapses:

    struct synapse_instance {
        size_type cv;
        std::vector<std::pair<unsigned, double>> param_value; // sorted vector of <param_id,value> pairs
        size_type target_index;
    };

    for (const auto& entry: cell.synapses()) {
        const std::string& name = entry.first;
        mechanism_info info = catalogue[name];
        std::vector<synapse_instance> sl;

        // Map from a param string identifier to a unique unsigned int identifier
        std::map<std::string, unsigned> param_map;

        // Map from a param unsigned int identifier to a default value
        std::map<unsigned, double> default_param_value;

        unsigned id=0;
        for (const auto& kv: info.parameters) {
            param_map[kv.first] = id;
            default_param_value[id++] = kv.second.default_value;
        }

        for (const placed<mechanism_desc>& pm: entry.second) {
            verify_mechanism(info, pm.item);

            synapse_instance in;

            auto param_value_map = default_param_value;
            for (const auto& kv: pm.item.values()) {
                param_value_map.at(param_map.at(kv.first)) = kv.second;
            }
            std::copy(param_value_map.begin(), param_value_map.end(), std::back_inserter(in.param_value));
            std::sort(in.param_value.begin(), in.param_value.end());

            in.target_index = pm.lid;
            in.cv = D.geometry.location_cv(cell_idx, pm.loc, cv_prefer::cv_nonempty);
            sl.push_back(std::move(in));
        }

        // Permute synapse instances so that they are in increasing order
        // (lexicographically) by CV, param_value set, and target, so that
        // instances in the same CV with the same parameter values are adjacent.
        // cv_order[i] is the index of the ith instance by this ordering.

        std::vector<size_type> cv_order;
        assign(cv_order, count_along(sl));
        sort_by(cv_order, [&](size_type i) {
            return std::tie(sl[i].cv, sl[i].param_value, sl[i].target_index);
        });

        bool coalesce = catalogue[name].linear && gprop.coalesce_synapses;

        fvm_mechanism_config config;
        config.kind = mechanismKind::point;
        for (auto& pentry: param_map) {
            config.param_values.emplace_back(pentry.first, std::vector<value_type>{});
        }

        const synapse_instance* prev = nullptr;
        for (auto i: cv_order) {
            const auto& in = sl[i];

            if (coalesce && prev && prev->cv==in.cv && prev->param_value==in.param_value) {
                ++config.multiplicity.back();
            }
            else {
                config.cv.push_back(in.cv);
                if (coalesce) {
                    config.multiplicity.push_back(1);
                }

                unsigned j = 0;
                for (auto& pentry: param_map) {
                    arb_assert(config.param_values[j].first==pentry.first);
                    auto it = std::lower_bound(in.param_value.begin(), in.param_value.end(), pentry.second, [](auto el, unsigned val) {return el.first < val;});

                    arb_assert(it!= in.param_value.end());
                    config.param_values[j++].second.push_back((*it).second);
                }
            }
            config.target.push_back(in.target_index);

            prev = &in;
        }

        // If synapse uses an ion, add to ion support.
        update_ion_support(info, config.cv);

        M.n_target += config.target.size();
        M.mechanisms[name] = std::move(config);
    }

    // Stimuli:

    if (!cell.stimuli().empty()) {
        const auto& stimuli = cell.stimuli();

        std::vector<size_type> stimuli_cv;
        assign_by(stimuli_cv, stimuli, [&D, cell_idx](auto& p) {
                return D.geometry.location_cv(cell_idx, p.loc, cv_prefer::cv_nonempty); });

        std::vector<size_type> cv_order;
        assign(cv_order, count_along(stimuli));
        sort_by(cv_order, [&](size_type i) { return stimuli_cv[i]; });

        fvm_mechanism_config config;
        config.kind = mechanismKind::point;
        // (param_values entries must be ordered by parameter name)
        config.param_values = {{"amplitude", {}}, {"delay", {}}, {"duration", {}}};

        for (auto i: cv_order) {
            config.cv.push_back(stimuli_cv[i]);
            config.param_values[0].second.push_back(stimuli[i].item.amplitude);
            config.param_values[1].second.push_back(stimuli[i].item.delay);
            config.param_values[2].second.push_back(stimuli[i].item.duration);
        }

        M.mechanisms["_builtin_stimulus"] = std::move(config);
    }

    // Ions:

    auto initial_ion_data_map = cell.region_assignments().get<initial_ion_data>();

    for (const auto& ion_cvs: ion_support) {
        const std::string& ion = ion_cvs.first;

        fvm_ion_config config;
        config.cv = ion_cvs.second;

        auto n_cv = config.cv.size();
        config.init_iconc.resize(n_cv);
        config.init_econc.resize(n_cv);
        config.reset_iconc.resize(n_cv);
        config.reset_econc.resize(n_cv);
        config.init_revpot.resize(n_cv);

        cable_cell_ion_data ion_data = *(value_by_key(dflt.ion_data, ion) | value_by_key(global_dflt.ion_data, ion));
        double dflt_iconc = ion_data.init_int_concentration;
        double dflt_econc = ion_data.init_ext_concentration;
        double dflt_rvpot = ion_data.init_reversal_potential;

        const mcable_map<initial_ion_data>& ion_on_cable = initial_ion_data_map[ion];

        auto pw_times = [](const pw_elements<double>& a, const pw_elements<double>& b) {
            return zip(a, b, [](double left, double right, pw_element<double> a, pw_element<double> b) { return a.second*b.second; });
        };

        for (auto i: count_along(config.cv)) {
            auto cv = config.cv[i];
            if (D.cv_area[cv]==0) continue;

            for (mcable c: D.geometry.cables(cv)) {
                auto iconc = pw_over_cable(ion_on_cable, c, dflt_iconc, [](auto x) { return x.initial.init_int_concentration; });
                auto econc = pw_over_cable(ion_on_cable, c, dflt_econc, [](auto x) { return x.initial.init_ext_concentration; });
                auto rvpot = pw_over_cable(ion_on_cable, c, dflt_rvpot, [](auto x) { return x.initial.init_reversal_potential; });

                config.reset_iconc[i] += embedding.integrate_area(c.branch, iconc);
                config.reset_econc[i] += embedding.integrate_area(c.branch, econc);
                config.init_revpot[i] += embedding.integrate_area(c.branch, rvpot);

                auto iconc_masked = pw_times(pw_over_cable(init_iconc_mask[ion], c, 1.), iconc);
                auto econc_masked = pw_times(pw_over_cable(init_econc_mask[ion], c, 1.), econc);

                config.init_iconc[i] += embedding.integrate_area(c.branch, iconc_masked);
                config.init_econc[i] += embedding.integrate_area(c.branch, econc_masked);
            }

            double oo_cv_area = 1./D.cv_area[cv];
            config.reset_iconc[i] *= oo_cv_area;
            config.reset_econc[i] *= oo_cv_area;
            config.init_revpot[i] *= oo_cv_area;
            config.init_iconc[i] *= oo_cv_area;
            config.init_econc[i] *= oo_cv_area;
        }

        M.ions[ion] = std::move(config);
    }

    std::unordered_map<std::string, mechanism_desc> revpot_tbl;
    std::unordered_set<std::string> revpot_specified;

    for (const auto& ion: util::keys(gprop.ion_species)) {
        if (auto maybe_revpot = value_by_key(dflt.reversal_potential_method, ion)
                              | value_by_key(global_dflt.reversal_potential_method, ion))
        {
            const mechanism_desc& revpot = *maybe_revpot;
            mechanism_info info = catalogue[revpot.name()];
            verify_mechanism(info, revpot);
            revpot_specified.insert(ion);

            bool writes_this_revpot = false;
            for (auto& iondep: info.ions) {
                if (iondep.second.write_reversal_potential) {
                    if (revpot_tbl.count(iondep.first)) {
                        auto& existing_revpot_desc = revpot_tbl.at(iondep.first);
                        if (existing_revpot_desc.name() != revpot.name() || existing_revpot_desc.values() != revpot.values()) {
                            throw cable_cell_error("inconsistent revpot ion assignment for mechanism "+revpot.name());
                        }
                    }
                    else {
                        revpot_tbl[iondep.first] = revpot;
                    }

                    writes_this_revpot |= iondep.first==ion;
                }
            }

            if (!writes_this_revpot) {
                throw cable_cell_error("revpot mechanism for ion "+ion+" does not write this reversal potential");
            }

            // Only instantiate if the ion is used.
            if (M.ions.count(ion)) {
                // Revpot mechanism already configured? Add cvs for this ion too.
                if (M.mechanisms.count(revpot.name())) {
                    fvm_mechanism_config& config = M.mechanisms[revpot.name()];
                    config.cv = unique_union(config.cv, M.ions[ion].cv);
                    config.norm_area.assign(config.cv.size(), 1.);

                    for (auto& pv: config.param_values) {
                        pv.second.assign(config.cv.size(), pv.second.front());
                    }
                }
                else {
                    fvm_mechanism_config config;
                    config.kind = mechanismKind::revpot;
                    config.cv = M.ions[ion].cv;
                    config.norm_area.assign(config.cv.size(), 1.);

                    std::map<std::string, double> param_value; // uses ordering of std::map
                    for (const auto& kv: info.parameters) {
                        param_value[kv.first] = kv.second.default_value;
                    }

                    for (auto& kv: revpot.values()) {
                        param_value[kv.first] = kv.second;
                    }

                    for (auto& kv: param_value) {
                        config.param_values.emplace_back(kv.first, std::vector<value_type>(config.cv.size(), kv.second));
                    }

                    M.mechanisms[revpot.name()] = std::move(config);
                }
            }
        }
    }

    // Confirm that all ions written to by a revpot have a corresponding entry in a reversal_potential_method table.
    for (auto& kv: revpot_tbl) {
        if (!revpot_specified.count(kv.first)) {
            throw cable_cell_error("revpot mechanism "+kv.second.name()+" also writes to ion "+kv.first);
        }
    }

    return M;
}

} // namespace arb
