#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <utility>
#include <memory>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/iexpr.hpp>
#include <arbor/math.hpp>
#include <arbor/morph/mcable_map.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/morphology.hpp>

#include "fvm_layout.hpp"
#include "threading/threading.hpp"
#include "util/maputil.hpp"
#include "util/meta.hpp"
#include "util/partition.hpp"
#include "util/piecewise.hpp"
#include "util/pw_over_cable.hpp"
#include "util/rangeutil.hpp"
#include "util/transform.hpp"
#include "util/unique.hpp"
#include "util/strprintf.hpp"

namespace arb {

using util::assign;
using util::assign_by;
using util::count_along;
using util::make_span;
using util::pw_elements;
using util::pw_element;
using util::pw_over_cable;
using util::sort;
using util::sort_by;
using util::stable_sort_by;
using util::value_by_key;

namespace {
template <typename V>
std::optional<V> operator|(const std::optional<V>& a, const std::optional<V>& b) {
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


// Building CV geometry
// --------------------
//
// CV geometry

cv_geometry::cv_geometry(const cable_cell& cell, const locset& ls):
        base(cell, ls)
{
    // Build location query map.
    auto n_cv = cv_parent.size();
    branch_cv_map.resize(1);
    std::vector<util::pw_elements<arb_size_type>>& bmap = branch_cv_map.back();
    for (auto cv: util::make_span(n_cv)) {
        for (auto cable: cables(cv)) {
            if (cable.branch>=bmap.size()) {
                bmap.resize(cable.branch+1);
            }

            // Ordering of CV ensures CV cables on any given branch are found sequentially.
            bmap[cable.branch].push_back(cable.prox_pos, cable.dist_pos, cv);
        }
    }
    cv_to_cell.assign(n_cv, 0);
    cell_cv_divs = {0, (arb_index_type)n_cv};
}

arb_size_type cv_geometry::location_cv(size_type cell_idx, const mlocation& loc, cv_prefer::type prefer) const {
    auto& pw_cv_offset = branch_cv_map.at(cell_idx).at(loc.branch);
    auto zero_extent = [&pw_cv_offset](auto j) {
        return pw_cv_offset.extent(j).first==pw_cv_offset.extent(j).second;
    };

    auto i = pw_cv_offset.index_of(loc.pos);
    auto i_max = pw_cv_offset.size()-1;
    auto cv_prox = pw_cv_offset.extent(i).first;

    // index_of() should have returned right-most matching interval.
    arb_assert(i==i_max || loc.pos<pw_cv_offset.extent(i+1).first);

    using namespace cv_prefer;
    switch (prefer) {
        case cv_distal:
            break;
        case cv_proximal:
            if (loc.pos==cv_prox && i>0) --i;
            break;
        case cv_nonempty:
            if (zero_extent(i)) {
                if (i>0 && !zero_extent(i-1)) --i;
                else if (i<i_max && !zero_extent(i+1)) ++i;
            }
            break;
        case cv_empty:
            if (loc.pos==cv_prox && i>0 && zero_extent(i-1)) --i;
            break;
    }

    index_type cv_base = cell_cv_divs.at(cell_idx);
    return cv_base+pw_cv_offset.value(i);
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

    template <typename Container>
    void append_divs(Container& left, const Container& right) {
        if (left.empty()) {
            left = right;
        }
        else if (!right.empty()) {
            append_offset(left, left.back(), tail(right));
        }
    };

}

// Merge CV geometry lists in-place.

ARB_ARBOR_API cv_geometry& append(cv_geometry& geom, const cv_geometry& right) {
    using util::append;
    using impl::tail;
    using impl::append_offset;
    using impl::append_divs;

    if (!right.n_cell()) {
        return geom;
    }

    if (!geom.n_cell()) {
        geom = right;
        return geom;
    }

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

ARB_ARBOR_API fvm_cv_discretization& append(fvm_cv_discretization& dczn, const fvm_cv_discretization& right) {
    using util::append;

    append(dczn.geometry, right.geometry);

    // Those in L and R: merge
    for (auto& [ion, data]: dczn.diffusive_ions) {
        const auto& rhs = right.diffusive_ions.find(ion);
        if (rhs != right.diffusive_ions.end()) {
            append(data.axial_inv_diffusivity, rhs->second.axial_inv_diffusivity);
            append(data.face_diffusivity, rhs->second.face_diffusivity);
        }
    }
    // Those only in R: add to L
    for (auto& [ion, data]: right.diffusive_ions) {
        const auto& lhs = dczn.diffusive_ions.find(ion);
        if (lhs == dczn.diffusive_ions.end()) {
            dczn.diffusive_ions[ion].axial_inv_diffusivity = data.axial_inv_diffusivity;
            dczn.diffusive_ions[ion].face_diffusivity      = data.face_diffusivity;
        }
    }

    append(dczn.face_conductance, right.face_conductance);
    append(dczn.cv_area, right.cv_area);
    append(dczn.cv_capacitance, right.cv_capacitance);
    append(dczn.init_membrane_potential, right.init_membrane_potential);
    append(dczn.temperature_K, right.temperature_K);
    append(dczn.diam_um, right.diam_um);

    append(dczn.axial_resistivity, right.axial_resistivity);

    return dczn;
}

// FVM discretization
// ------------------

ARB_ARBOR_API fvm_cv_discretization fvm_cv_discretize(const cable_cell& cell, const cable_cell_parameter_set& global_dflt) {
    const auto& dflt = cell.default_parameters();
    fvm_cv_discretization D;

    D.geometry = cv_geometry(cell,
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

    const auto& assignments = cell.region_assignments();
    const auto& resistivity = assignments.get<axial_resistivity>();
    const auto& capacitance = assignments.get<membrane_capacitance>();
    const auto& potential   = assignments.get<init_membrane_potential>();
    const auto& temperature = assignments.get<temperature_K>();
    const auto& diffusivity = assignments.get<ion_diffusivity>();

    // Set up for ion diffusivity
    std::unordered_map<std::string, mcable_map<double>> inverse_diffusivity;
    std::unordered_map<std::string, fvm_diffusion_info> diffusive_ions;

    // Collect all eglible ions: those where any cable has finite diffusivity
    for (const auto& [ion, data]: global_dflt.ion_data) {
        if (data.diffusivity.value_or(0.0) != 0.0) {
            diffusive_ions[ion] = {};
        }
    }
    for (const auto& [ion, data]: dflt.ion_data) {
        if (data.diffusivity.value_or(0.0) != 0.0) {
            diffusive_ions[ion] = {};
        }
    }
    for (const auto& [ion, data]: diffusivity) {
        auto diffusive = std::any_of(data.begin(),
                                     data.end(),
                                     [](const auto& kv) { return kv.second.value != 0.0; });
        if (diffusive) {
            diffusive_ions[ion] = {};
        }
    }

    // Remap diffusivity to resistivity
    for (auto& [ion, data]: diffusive_ions) {
        auto& id_map = inverse_diffusivity[ion];
        // Treat specific assignments
        if (auto map = diffusivity.find(ion); map != diffusivity.end()) {
            for (const auto& [k, v]: map->second) {
                if (v.value <= 0.0) {
                    throw cable_cell_error{util::pprintf("Illegal diffusivity '{}' for ion '{}' at '{}'.", v.value, ion, k)};
                }
                id_map.insert(k, 1.0/v.value);
            }
        }
        arb_value_type def = 0.0;
        if (auto data = value_by_key(global_dflt.ion_data, ion);
            data && data->diffusivity) {
            def = data->diffusivity.value();
        }
        if (auto data = value_by_key(dflt.ion_data, ion);
            data && data->diffusivity) {
            def = data->diffusivity.value();
        }
        if (def <= 0.0) {
            throw cable_cell_error{util::pprintf("Illegal global diffusivity '{}' for ion '{}'; possibly unset."
                                                 " Please define a positive global or cell default.", def, ion)};
        }

        // Write inverse diffusivity / diffuse resistivity map
        auto& id = data.axial_inv_diffusivity;
        id.resize(1);
        msize_t n_branch = D.geometry.n_branch(0);
        id.reserve(n_branch);
        for (msize_t i = 0; i<n_branch; ++i) {
            auto pw = pw_over_cable(id_map, mcable{i, 0., 1.}, 1.0/def);
            id[0].push_back(pw);
        }
        // Prepare conductivity map
        data.face_diffusivity.resize(n_cv);
    }

    D.axial_resistivity.resize(1);
    msize_t n_branch = D.geometry.n_branch(0);
    auto& ax_res_0 = D.axial_resistivity[0];
    ax_res_0.reserve(n_branch);
    for (msize_t i = 0; i<n_branch; ++i) {
        ax_res_0.emplace_back(pw_over_cable(resistivity, mcable{i, 0., 1.}, dflt_resistivity));
    }

    const auto& embedding = cell.embedding();
    for (auto i: count_along(D.geometry.cv_parent)) {
        auto cv_cables = D.geometry.cables(i);
        // Computing face_conductance and face_diffusivity
        //
        // Flux between adjacent CVs is computed as if there were no membrane currents, and with the CV voltage
        // values taken to be exact at a reference point in each CV:
        //     * If the CV is unbranched, the reference point is taken to be the CV midpoint.
        //     * If the CV is branched, the reference point is taken to be closest branch point to
        //       the interface between the two CVs.
        D.face_conductance[i] = 0;
        for (auto& [ion, info]: diffusive_ions) {
            info.face_diffusivity[i] = 0.0;
        }

        arb_index_type p = D.geometry.cv_parent[i];
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
            for (auto& [ion, info]: diffusive_ions) {
                double resistance = embedding.integrate_ixa(span, info.axial_inv_diffusivity[0].at(bid));
                info.face_diffusivity[i] = 1.0/resistance; // scale to m^2/s
            }
        }

        D.cv_area[i] = 0;
        D.cv_capacitance[i] = 0;
        D.init_membrane_potential[i] = 0;
        D.diam_um[i] = 0;
        double cv_length = 0;

        for (mcable c: cv_cables) {
            D.cv_area[i]                 += embedding.integrate_area(c);
            D.cv_capacitance[i]          += embedding.integrate_area(c.branch, pw_over_cable(capacitance, c, dflt_capacitance));
            D.init_membrane_potential[i] += embedding.integrate_area(c.branch, pw_over_cable(potential,   c, dflt_potential));
            D.temperature_K[i]           += embedding.integrate_area(c.branch, pw_over_cable(temperature, c, dflt_temperature));
            cv_length += embedding.integrate_length(c);
        }

        if (D.cv_area[i]>0) {
            auto A = D.cv_area[i];
            D.init_membrane_potential[i] /= A;
            D.temperature_K[i] /= A;

            for (auto& [ion, info]: diffusive_ions) {
                info.face_diffusivity[i] /= A;
            }
            // If parent is trivial, and there is no grandparent, then we can use values from this CV
            // to get initial values for the parent. (The other case, when there is a grandparent, is
            // caught below.)
            if (p!=-1 && D.geometry.cv_parent[p]==-1 && D.cv_area[p]==0) {
                D.init_membrane_potential[p] = D.init_membrane_potential[i];
                D.temperature_K[p] = D.temperature_K[i];
            }
        }
        else if (p!=-1) {
            // Use parent CV to get a sensible initial value for voltage and temp on zero-size CVs.
            D.init_membrane_potential[i] = D.init_membrane_potential[p];
            D.temperature_K[i] = D.temperature_K[p];
        }

        if (cv_length>0) {
            D.diam_um[i] = D.cv_area[i]/(cv_length*math::pi<double>);
        }
    }

    D.diffusive_ions = std::move(diffusive_ions);
    return D;
}

ARB_ARBOR_API fvm_cv_discretization fvm_cv_discretize(const std::vector<cable_cell>& cells,
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

// Voltage interpolation
// ---------------------
//
// Interpolated voltages and axial current at a given site are determined
// from 'voltage references'. A voltage reference is a CV from which the
// membrane voltage is taken, and a location within that CV where the
// voltage is deemed to be accurate.
//
// A CV that includes no fork points has one reference location which is
// the centre of the CV (by branch length). Otherwise, every fork in a CV
// is regarded as being a reference location.
//
// Voltage references should comprise adjacent CVs, however should the site
// lie between fork points within the one CV, there is nothing to interpolate
// and the voltage references will all come from the one CV containing the
// site.

struct voltage_reference {
    arb_index_type cv = -1;
    mlocation loc;
};

struct voltage_reference_pair {
    voltage_reference proximal;
    voltage_reference distal;
};

// Collection of other locations that are coincident under projection.
std::vector<mlocation> coincident_locations(const morphology& m, const mlocation& x) {
    std::vector<mlocation> result;
    if (x.pos==0) {
        msize_t parent_bid = m.branch_parent(x.branch);
        if (parent_bid!=mnpos) {
            result.push_back({parent_bid, 1});
        }
        for (msize_t sibling_bid: m.branch_children(parent_bid)) {
            if (sibling_bid!=x.branch) {
                result.push_back({sibling_bid, 0});
            }
        }
    }
    else if (x.pos==1) {
        for (msize_t child_bid: m.branch_children(x.branch)) {
            result.push_back({child_bid, 0});
        }
    }
    return result;
}

// Test if location intersects (sorted) sequence of cables.
template <typename Seq>
bool cables_intersect_location(Seq&& cables, const mlocation& x) {
    struct cmp_branch {
        bool operator()(const mcable& c, msize_t bid) const { return c.branch<bid; }
        bool operator()(msize_t bid, const mcable& c) const { return bid<c.branch; }
    };

    using std::begin;
    using std::end;
    auto eqr = std::equal_range(begin(cables), end(cables), x.branch, cmp_branch{});

    return util::any_of(util::make_range(eqr),
        [&x](const mcable& c) { return c.prox_pos<=x.pos && x.pos<=c.dist_pos; });
}

voltage_reference_pair fvm_voltage_reference_points(const morphology& morph, const cv_geometry& geom, arb_size_type cell_idx, const mlocation& site) {
    voltage_reference site_ref, parent_ref, child_ref;
    bool check_parent = true, check_child = true;
    msize_t bid = site.branch;

    // 'Simple' CVs contain no fork points, and are represented by a single cable.
    auto cv_simple = [&geom](auto cv) { return geom.cables(cv).size()==1u; };

    auto cv_midpoint = [&geom](auto cv) {
        // Under assumption that CV is simple:
        mcable c = geom.cables(cv).front();
        return mlocation{c.branch, (c.prox_pos+c.dist_pos)/2};
    };

    auto cv_contains_fork = [&](auto cv, const mlocation& x) {
        // CV contains fork if it intersects any location coincident with x
        // other than x itselfv.

        if (cv_simple(cv)) return false;
        auto locs = coincident_locations(morph, x);

        return util::any_of(locs, [&](mlocation y) { return cables_intersect_location(geom.cables(cv), y); });
    };

    site_ref.cv = geom.location_cv(cell_idx, site, cv_prefer::cv_empty);
    if (cv_simple(site_ref.cv)) {
        site_ref.loc = cv_midpoint(site_ref.cv);
    }
    else if (cv_contains_fork(site_ref.cv, mlocation{bid, 0})) {
        site_ref.loc = mlocation{bid, 0};
        check_parent = false;
    }
    else {
        // CV not simple, and without head of branch as fork point, must contain
        // tail of branch as a fork point.
        arb_assert(cv_contains_fork(site_ref.cv, mlocation{bid, 1}));

        site_ref.loc = mlocation{bid, 1};
        check_child = false;
    }

    if (check_parent) {
        parent_ref.cv = geom.cv_parent[site_ref.cv];
    }
    if (parent_ref.cv!=-1) {
        parent_ref.loc = cv_simple(parent_ref.cv)? cv_midpoint(parent_ref.cv): mlocation{bid, 0};
        arb_assert(parent_ref.loc.branch==bid);
    }

    if (check_child) {
        for (auto child_cv: geom.children(site_ref.cv)) {
            mcable child_prox_cable = geom.cables(child_cv).front();
            if (child_prox_cable.branch==bid) {
                child_ref.cv = child_cv;
                break;
            }
        }
    }
    if (child_ref.cv!=-1) {
        child_ref.loc = cv_simple(child_ref.cv)? cv_midpoint(child_ref.cv): mlocation{bid, 1};
        arb_assert(child_ref.loc.branch==bid);
    }

    // If both child and parent references are possible, pick based on distallity with respect
    // to the site_ref location.

    if (child_ref.cv!=-1 && parent_ref.cv!=-1) {
        if (site.pos<site_ref.loc.pos) child_ref.cv = -1; // i.e. use parent.
        else parent_ref.cv = -1; // i.e. use child.
    }

    voltage_reference_pair result;
    if (child_ref.cv!=-1) {
        result.proximal = site_ref;
        result.distal = child_ref;
    }
    else if (parent_ref.cv!=-1) {
        result.proximal = parent_ref;
        result.distal = site_ref;
    }
    else {
        result.proximal = site_ref;
        result.distal = site_ref;
    }

    return result;
}

// Interpolate membrane voltage from reference points in adjacent CVs.

ARB_ARBOR_API fvm_voltage_interpolant fvm_interpolate_voltage(const cable_cell& cell, const fvm_cv_discretization& D, arb_size_type cell_idx, const mlocation& site) {
    auto& embedding = cell.embedding();
    fvm_voltage_interpolant vi;

    auto vrefs = fvm_voltage_reference_points(cell.morphology(), D.geometry, cell_idx, site);
    vi.proximal_cv = vrefs.proximal.cv;
    vi.distal_cv = vrefs.distal.cv;

    arb_assert(vrefs.proximal.loc.branch==site.branch);
    arb_assert(vrefs.distal.loc.branch==site.branch);

    if (vrefs.proximal.cv==vrefs.distal.cv) { // (no interpolation)
        vi.proximal_coef = 1.0;
        vi.distal_coef = 0.0;
    }
    else {
        msize_t bid = site.branch;

        arb_assert(vrefs.proximal.loc.pos<vrefs.distal.loc.pos);
        mcable rr_span  = mcable{bid, vrefs.proximal.loc.pos , vrefs.distal.loc.pos};
        double rr_resistance = embedding.integrate_ixa(rr_span, D.axial_resistivity[0].at(bid));

        // Note: site is not necessarily distal to the most proximal reference point.
        bool flip_rs = vrefs.proximal.loc.pos>site.pos;
        mcable rs_span = flip_rs? mcable{bid, site.pos, vrefs.proximal.loc.pos}
                                : mcable{bid, vrefs.proximal.loc.pos, site.pos};

        double rs_resistance = embedding.integrate_ixa(rs_span, D.axial_resistivity[0].at(bid));
        if (flip_rs) {
            rs_resistance = -rs_resistance;
        }

        double p = rs_resistance/rr_resistance;
        vi.proximal_coef = 1-p;
        vi.distal_coef = p;
    }
    return vi;
}

// Axial current as linear combination of membrane voltages at reference points in adjacent CVs.

ARB_ARBOR_API fvm_voltage_interpolant fvm_axial_current(const cable_cell& cell, const fvm_cv_discretization& D, arb_size_type cell_idx, const mlocation& site) {
    auto& embedding = cell.embedding();
    fvm_voltage_interpolant vi;

    auto vrefs = fvm_voltage_reference_points(cell.morphology(), D.geometry, cell_idx, site);
    vi.proximal_cv = vrefs.proximal.cv;
    vi.distal_cv = vrefs.distal.cv;

    if (vi.proximal_cv==vi.distal_cv) {
        vi.proximal_coef = 0;
        vi.distal_coef = 0;
    }
    else {
        msize_t bid = site.branch;

        arb_assert(vrefs.proximal.loc.pos<vrefs.distal.loc.pos);
        mcable rr_span  = mcable{bid, vrefs.proximal.loc.pos , vrefs.distal.loc.pos};
        double rr_conductance = 100/embedding.integrate_ixa(rr_span, D.axial_resistivity[cell_idx].at(bid)); // [µS]

        vi.proximal_coef = rr_conductance;
        vi.distal_coef = -rr_conductance;
    }

    return vi;
}

// FVM mechanism data
// ------------------

// CVs are absolute (taken from combined discretization) so do not need to be shifted.
// Only target numbers need to be shifted.

fvm_mechanism_data& append(fvm_mechanism_data& left, const fvm_mechanism_data& right) {
    using util::append;
    using impl::append_offset;
    using impl::append_divs;

    arb_size_type target_offset = left.n_target;

    for (const auto& [k, R]: right.ions) {
        fvm_ion_config& L = left.ions[k];

        append(L.cv, R.cv);
        append(L.init_iconc, R.init_iconc);
        append(L.init_econc, R.init_econc);
        append(L.reset_iconc, R.reset_iconc);
        append(L.reset_econc, R.reset_econc);
        append(L.init_revpot, R.init_revpot);
        append(L.face_diffusivity, R.face_diffusivity);
        L.is_diffusive |= R.is_diffusive;
        L.econc_written  |= R.econc_written;
        L.iconc_written  |= R.iconc_written;
        L.revpot_written |= R.revpot_written;
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
            append(L.peer_cv, R.peer_cv);
            append(L.multiplicity, R.multiplicity);
            append(L.norm_area, R.norm_area);
            append(L.local_weight, R.local_weight);
            append_offset(L.target, target_offset, R.target);

            arb_assert(util::equal(L.param_values, R.param_values,
                [](auto& a, auto& b) { return a.first==b.first; }));
            arb_assert(L.param_values.size()==R.param_values.size());

            for (auto j: count_along(R.param_values)) {
                arb_assert(L.param_values[j].first==R.param_values[j].first);
                append(L.param_values[j].second, R.param_values[j].second);
            }
        }
    }

    append(left.stimuli.cv, right.stimuli.cv);
    append(left.stimuli.cv_unique, right.stimuli.cv_unique);
    append(left.stimuli.frequency, right.stimuli.frequency);
    append(left.stimuli.phase, right.stimuli.phase);
    append(left.stimuli.envelope_time, right.stimuli.envelope_time);
    append(left.stimuli.envelope_amplitude, right.stimuli.envelope_amplitude);

    left.n_target += right.n_target;
    left.post_events |= right.post_events;

    append_divs(left.target_divs, right.target_divs);
    arb_assert(left.n_target==left.target_divs.back());

    return left;
}

ARB_ARBOR_API std::unordered_map<cell_member_type, arb_size_type> fvm_build_gap_junction_cv_map(
    const std::vector<cable_cell>& cells,
    const std::vector<cell_gid_type>& gids,
    const fvm_cv_discretization& D)
{
    arb_assert(cells.size() == gids.size());
    std::unordered_map<cell_member_type, arb_size_type> gj_cvs;
    for (auto cell_idx: util::make_span(0, cells.size())) {
        for (const auto& mech : cells[cell_idx].junctions()) {
            for (const auto& gj: mech.second) {
                gj_cvs.insert({cell_member_type{gids[cell_idx], gj.lid}, D.geometry.location_cv(cell_idx, gj.loc, cv_prefer::cv_nonempty)});
            }
        }
    }
    return gj_cvs;
}

ARB_ARBOR_API std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> fvm_resolve_gj_connections(
    const std::vector<cell_gid_type>& gids,
    const cell_label_range& gj_data,
    const std::unordered_map<cell_member_type, arb_size_type>& gj_cvs,
    const recipe& rec)
{
    // Construct and resolve all gj_connections.
    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns;
    label_resolution_map resolution_map({gj_data, gids});
    auto gj_resolver = resolver(&resolution_map);
    for (const auto& gid: gids) {
        std::vector<fvm_gap_junction> local_conns;
        for (const auto& conn: rec.gap_junctions_on(gid)) {
            auto local_idx = gj_resolver.resolve({gid, conn.local});
            auto peer_idx  = gj_resolver.resolve(conn.peer);

            auto local_cv = gj_cvs.at({gid, local_idx});
            auto peer_cv  = gj_cvs.at({conn.peer.gid, peer_idx});

            local_conns.push_back({local_idx, local_cv, peer_cv, conn.weight});
        }
        // Sort local_conns by local_cv.
        util::sort(local_conns);
        gj_conns[gid] = std::move(local_conns);
    }
    return gj_conns;
}

std::unordered_map<std::string, fvm_mechanism_config>
make_voltage_mechanism_config(const cable_cell_global_properties& gprop,
                              const region_assignment<voltage_process> assignments,
                              const mechanism_catalogue& catalogue,
                              iexpr_ptr unit_scale,
                              unsigned cell_idx,
                              const fvm_cv_discretization& D,
                              const concrete_embedding& embedding,
                              const mprovider& provider);

fvm_mechanism_data fvm_build_mechanism_data(
    const cable_cell_global_properties& gprop,
    const cable_cell& cell,
    const std::vector<fvm_gap_junction>& gj_conns,
    const fvm_cv_discretization& D,
    arb_size_type cell_idx);

ARB_ARBOR_API fvm_mechanism_data fvm_build_mechanism_data(
    const cable_cell_global_properties& gprop,
    const std::vector<cable_cell>& cells,
    const std::vector<cell_gid_type>& gids,
    const std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>>& gj_conns,
    const fvm_cv_discretization& D,
    const execution_context& ctx)
{
    std::vector<fvm_mechanism_data> cell_mech(cells.size());
    threading::parallel_for::apply(0, cells.size(), ctx.thread_pool.get(), [&] (int i) {
        cell_mech[i] = fvm_build_mechanism_data(gprop, cells[i], gj_conns.at(gids[i]), D, i);
    });

    fvm_mechanism_data combined;
    for (auto cell_idx: count_along(cells)) {
        append(combined, cell_mech[cell_idx]);
    }
    return combined;
}


// Verify mechanism ion usage, parameter values.
void verify_mechanism(const cable_cell_global_properties& gprop,
                      const fvm_cv_discretization& D,
                      const mechanism_info& info,
                      const mechanism_desc& desc) {
    const auto& global_ions = gprop.ion_species;

    for (const auto& pv: desc.values()) {
        if (!info.parameters.count(pv.first)) {
            throw no_such_parameter(desc.name(), pv.first);
        }
        if (!info.parameters.at(pv.first).valid(pv.second)) {
            throw invalid_parameter_value(desc.name(), pv.first, pv.second);
        }
    }

    for (const auto& [ion, dep]: info.ions) {
        if (!global_ions.count(ion)) {
            throw cable_cell_error(
                "mechanism "+desc.name()+" uses ion "+ion+ " which is missing in global properties");
        }

        if (dep.verify_ion_charge) {
            if (dep.expected_ion_charge!=global_ions.at(ion)) {
                throw cable_cell_error(
                    "mechanism "+desc.name()+" uses ion "+ion+ " expecting a different valence");
            }
        }

        if (dep.write_reversal_potential && (dep.write_concentration_int || dep.write_concentration_ext)) {
            throw cable_cell_error("mechanism "+desc.name()+" writes both reversal potential and concentration");
        }

        auto is_diffusive = D.diffusive_ions.count(ion);
        if (dep.access_concentration_diff && !is_diffusive) {
            throw illegal_diffusive_mechanism(desc.name(), ion);
        }
    }
}

// Construct FVM mechanism data for a single cell.

fvm_mechanism_data fvm_build_mechanism_data(
    const cable_cell_global_properties& gprop,
    const cable_cell& cell,
    const std::vector<fvm_gap_junction>& gj_conns,
    const fvm_cv_discretization& D,
    arb_size_type cell_idx)
{
    using size_type = arb_size_type;
    using index_type = arb_index_type;
    using value_type = arb_value_type;

    const auto& catalogue = gprop.catalogue;
    const auto& embedding = cell.embedding();
    const auto& provider  = cell.provider();

    const auto& global_dflt = gprop.default_parameters;
    const auto& dflt = cell.default_parameters();

    std::unordered_set<std::string> write_xi;
    std::unordered_set<std::string> write_xo;

    fvm_mechanism_data M;

    const auto& assignments = cell.region_assignments();

    // Track ion usage of mechanisms so that ions are only instantiated where required.
    std::unordered_map<std::string, std::vector<index_type>> ion_support;

    // add diffusive ions to support: If diffusive, it's everywhere.
    for (const auto& [ion, data]: D.diffusive_ions) {
        auto& s = ion_support[ion];
        s.resize(D.geometry.size());
        std::iota(s.begin(), s.end(), 0);
    }

    auto update_ion_support = [&ion_support](const mechanism_info& info, const std::vector<index_type>& cvs) {
        arb_assert(util::is_sorted(cvs));

        for (const auto& ion: util::keys(info.ions)) {
            auto& support = ion_support[ion];
            support = unique_union(support, cvs);
        }
    };

    std::unordered_map<std::string, mcable_map<double>> init_iconc_mask;
    std::unordered_map<std::string, mcable_map<double>> init_econc_mask;
    iexpr_ptr unit_scale = thingify(iexpr::scalar(1.0), provider);

    // Voltage mechanisms

    {
        auto assigns = assignments.get<voltage_process>();
        auto configs = make_voltage_mechanism_config(gprop,
                                                     assigns,
                                                     catalogue,
                                                     unit_scale,
                                                     cell_idx,
                                                     D,
                                                     embedding,
                                                     provider);
        M.mechanisms.insert(configs.begin(), configs.end());
    }
    // Density mechanisms:
    for (const auto& [name, cables]: assignments.get<density>()) {
        mechanism_info info = catalogue[name];

        fvm_mechanism_config config;
        if (info.kind != arb_mechanism_kind_density) {
            throw cable_cell_error("expected density mechanism, got " +name +" which has " +arb_mechanism_kind_str(info.kind));
        }
        config.kind = arb_mechanism_kind_density;

        std::vector<std::string> param_names;
        assign(param_names, util::keys(info.parameters));
        sort(param_names);

        std::vector<double> param_dflt;
        std::size_t n_param = param_names.size();
        param_dflt.reserve(n_param);
        config.param_values.reserve(n_param);

        for (std::size_t i = 0; i<n_param; ++i) {
            const auto& p = param_names[i];
            config.param_values.emplace_back(p, std::vector<value_type>{});
            param_dflt.push_back(info.parameters.at(p).default_value);
        }

        mcable_map<double> support;
        std::vector<mcable_map<std::pair<double, iexpr_ptr>>> param_maps;

        param_maps.resize(n_param);

        for (const auto& [cable, density_iexpr]: cables) {
            const auto& mech = density_iexpr.first.mech;
            const auto& scale_expr = density_iexpr.second;

            verify_mechanism(gprop, D, info, mech);
            const auto& set_params = mech.values();

            support.insert(cable, 1.);
            for (std::size_t i = 0; i<n_param; ++i) {
                double value = value_by_key(set_params, param_names[i]).value_or(param_dflt[i]);
                auto scale_it = scale_expr.find(param_names[i]);
                param_maps[i].insert(cable, {value, scale_it == scale_expr.end()
                                                        ? unit_scale : scale_it->second});
            }
        }

        std::vector<double> param_on_cv(n_param);

        for (auto cv: D.geometry.cell_cvs(cell_idx)) {
            double area = 0;
            util::fill(param_on_cv, 0.);

            for (mcable c: D.geometry.cables(cv)) {
                double area_on_cable = embedding.integrate_area(c.branch, pw_over_cable(support, c, 0.));
                if (!area_on_cable) continue;

                area += area_on_cable;
                for (std::size_t i = 0; i<n_param; ++i) {
                  param_on_cv[i] += embedding.integrate_area(
                      c.branch,
                      pw_over_cable(
                          param_maps[i], c, 0., [&](const mcable &c, const auto& x) {
                              return x.first * x.second->eval(provider, c);
                          }));
                }
            }

            if (area>0) {
                config.cv.push_back(cv);
                config.norm_area.push_back(area/D.cv_area[cv]);

                double oo_area = 1./area;
                for (auto i: count_along(param_on_cv)) {
                    config.param_values[i].second.push_back(param_on_cv[i]*oo_area);
                }
            }
        }

        for (const auto& [ion, dep]: info.ions) {
            if (dep.write_concentration_int) {
                write_xi.insert(ion);
                for (auto c: support) {
                    bool ok = init_iconc_mask[ion].insert(c.first, 0.);
                    if (!ok) {
                        throw cable_cell_error("overlapping ion concentration writing mechanism "+name);
                    }
                }
            }
            if (dep.write_concentration_ext) {
                write_xo.insert(ion);
                for (auto c: support) {
                    bool ok = init_econc_mask[ion].insert(c.first, 0.);
                    if (!ok) {
                        throw cable_cell_error("overlapping ion concentration writing mechanism "+name);
                    }
                }
            }
        }

        update_ion_support(info, config.cv);
        if (!config.cv.empty()) M.mechanisms[name] = std::move(config);
    }

    // Synapses:

    struct synapse_instance {
        size_type cv;
        std::size_t param_values_offset;
        size_type target_index;
    };

    // Working vectors for synapse collation:
    std::vector<double> default_param_value;
    std::vector<double> all_param_values;
    std::vector<synapse_instance> inst_list;
    std::vector<size_type> cv_order;

    bool post_events = false;

    for (const auto& entry: cell.synapses()) {
        const std::string& name = entry.first;
        mechanism_info info = catalogue[name];

        if (info.kind != arb_mechanism_kind_point) {
            throw cable_cell_error("expected point mechanism, got " +name +" which has " +arb_mechanism_kind_str(info.kind));
        }

        post_events |= info.post_events;
        std::size_t n_param = info.parameters.size();
        std::size_t n_inst = entry.second.size();

        default_param_value.resize(n_param);
        inst_list.clear();
        inst_list.reserve(n_inst);

        all_param_values.resize(n_param*n_inst);

        // Vectors of parameter values are stored in the order of
        // parameters given by info.parameters. param_index holds
        // the mapping from parameter names to their index in this
        // order.

        std::unordered_map<std::string, unsigned> param_index;

        unsigned ix=0;
        for (const auto& kv: info.parameters) {
            param_index[kv.first] = ix;
            default_param_value.at(ix++) = kv.second.default_value;
        }
        arb_assert(ix==n_param);

        std::size_t offset = 0;
        for (const placed<synapse>& pm: entry.second) {
            const auto& mech = pm.item.mech;
            verify_mechanism(gprop, D, info, mech);

            synapse_instance in{};

            in.param_values_offset = offset;
            offset += n_param;
            arb_assert(offset<=all_param_values.size());

            double* in_param = all_param_values.data()+in.param_values_offset;
            std::copy(default_param_value.begin(), default_param_value.end(), in_param);

            for (const auto& kv: mech.values()) {
                in_param[param_index.at(kv.first)] = kv.second;
            }

            in.target_index = pm.lid;
            in.cv = D.geometry.location_cv(cell_idx, pm.loc, cv_prefer::cv_nonempty);
            inst_list.push_back(std::move(in));
        }

        // Permute synapse instances so that they are in increasing order
        // (lexicographically) by CV, param_value set, and target, so that
        // instances in the same CV with the same parameter values are adjacent.
        // cv_order[i] is the index of the ith instance by this ordering.

        auto cmp_inst_param = [n_param, &all_param_values](const synapse_instance& a, const synapse_instance& b) {
            const double* aparam = all_param_values.data()+a.param_values_offset;
            const double* bparam = all_param_values.data()+b.param_values_offset;

            for (auto j: make_span(n_param)) {
                if (aparam[j]<bparam[j]) return -1;
                if (bparam[j]<aparam[j]) return 1;
            }
            return 0;
        };

        assign(cv_order, count_along(inst_list));
        sort(cv_order, [&](size_type i, size_type j) {
            const synapse_instance& a = inst_list[i];
            const synapse_instance& b = inst_list[j];

            if (a.cv<b.cv) return true;
            if (b.cv<a.cv) return false;

            auto cmp_param = cmp_inst_param(a, b);
            if (cmp_param<0) return true;
            if (cmp_param>0) return false;

            // CV and all parameters are equal, so finally sort on target index.
            return a.target_index<b.target_index;
        });

        bool coalesce = !info.random_variables.size() && info.linear && gprop.coalesce_synapses;

        fvm_mechanism_config config;
        config.kind = arb_mechanism_kind_point;
        for (auto& kv: info.parameters) {
            config.param_values.emplace_back(kv.first, std::vector<value_type>{});
            if (!coalesce) {
                config.param_values.back().second.reserve(n_inst);
            }
        }

        const synapse_instance* prev = nullptr;
        for (auto i: cv_order) {
            const auto& in = inst_list[i];

            if (coalesce && prev && prev->cv==in.cv && cmp_inst_param(*prev, in)==0) {
                ++config.multiplicity.back();
            }
            else {
                config.cv.push_back(in.cv);
                if (coalesce) {
                    config.multiplicity.push_back(1);
                }

                for (auto j: make_span(n_param)) {
                    config.param_values[j].second.push_back(all_param_values[in.param_values_offset+j]);
                }
            }
            config.target.push_back(in.target_index);

            prev = &in;
        }

        // If synapse uses an ion, add to ion support.
        update_ion_support(info, config.cv);

        for (const auto& [ion, dep]: info.ions) {
            if (dep.write_concentration_int) {
                write_xi.insert(ion);
            }
            if (dep.write_concentration_ext) {
                write_xo.insert(ion);
            }
        }

        M.n_target += config.target.size();
        if (!config.cv.empty()) M.mechanisms[name] = std::move(config);
    }
    M.post_events = post_events;

    // Gap junctions:

    struct junction_desc {
        std::string name;                     // mechanism name.
        std::vector<value_type> param_values; // overridden parameter values.
    };

    // Gap-junction mechanisms are handled differently from point mechanisms.
    // There is a separate mechanism instance at the local site of every gap-junction connection,
    // meaning there can be multiple gap-junction mechanism instances of the same type (name) per
    // lid.
    // As a result, building fvm_mechanism_config per junction mechanism is split into 2 phases.
    // (1) For every type (name) of gap-junction mechanism used on the cell, an fvm_mechanism_config
    //     object is constructed and only the kind and parameter names are set. The object is
    //     stored in the `junction_configs` map. Another map `lid_junction_desc` containing the
    //     name and parameter values of the mechanism per lid is stored, needed to complete the
    //     description of the fvm_mechanism_config object in the next step.
    // (2) For every gap-junction connection, the cv, peer_cv, local_weight and parameter values
    //     of the mechanism present on the local lid of the connection are added to the
    //     fvm_mechanism_config of that mechanism. This completes the fvm_mechanism_config
    //     description for each gap-junction mechanism.

    std::unordered_map<std::string, fvm_mechanism_config> junction_configs;
    std::unordered_map<cell_lid_type, junction_desc> lid_junction_desc;
    for (const auto& [name, placements]: cell.junctions()) {
        mechanism_info info = catalogue[name];
        if (info.kind != arb_mechanism_kind_gap_junction) {
            throw cable_cell_error("expected gap_junction mechanism, got " +name +" which has " +arb_mechanism_kind_str(info.kind));
        }

        fvm_mechanism_config config;
        config.kind = arb_mechanism_kind_gap_junction;

        std::vector<std::string> param_names;
        std::vector<double> param_dflt;

        assign(param_names, util::keys(info.parameters));
        std::size_t n_param = param_names.size();

        param_dflt.reserve(n_param);
        for (const auto& p: param_names) {
            config.param_values.emplace_back(p, std::vector<value_type>{});
            param_dflt.push_back(info.parameters.at(p).default_value);
        }

        for (const placed<junction>& pm: placements) {
            const auto& mech = pm.item.mech;
            verify_mechanism(gprop, D, info, mech);
            const auto& set_params = mech.values();

            junction_desc per_lid;
            per_lid.name = name;
            for (std::size_t i = 0; i<n_param; ++i) {
                per_lid.param_values.push_back(value_by_key(set_params, param_names[i]).value_or(param_dflt[i]));
            }
            lid_junction_desc.insert({pm.lid, std::move(per_lid)});
        }

        for (const auto& [ion, dep]: info.ions) {
            if (dep.write_concentration_int) {
                write_xi.insert(ion);
            }
            if (dep.write_concentration_ext) {
                write_xo.insert(ion);
            }
        }

        junction_configs[name] = std::move(config);
    }

    // Iterate over the gj_conns local to the cell, and complete the fvm_mechanism_config.
    // The gj_conns are expected to be sorted by local CV index.
    for (const auto& conn: gj_conns) {
        auto local_junction_desc = lid_junction_desc[conn.local_idx];
        auto& config = junction_configs[local_junction_desc.name];

        config.cv.push_back(conn.local_cv);
        config.peer_cv.push_back(conn.peer_cv);
        config.local_weight.push_back(conn.weight);
        for (unsigned i = 0; i < local_junction_desc.param_values.size(); ++i) {
            config.param_values[i].second.push_back(local_junction_desc.param_values[i]);
        }
    }

    // Add non-empty fvm_mechanism_config to the fvm_mechanism_data
    for (auto [name, config]: junction_configs) {
        if (!config.cv.empty()) M.mechanisms[name] = std::move(config);
    }

    // Stimuli:

    if (!cell.stimuli().empty()) {
        const auto& stimuli = cell.stimuli();
        fvm_stimulus_config config;

        std::vector<size_type> stimuli_cv;
        assign_by(stimuli_cv, stimuli, [&D, cell_idx](auto& p) {
                return D.geometry.location_cv(cell_idx, p.loc, cv_prefer::cv_nonempty); });

        std::vector<size_type> cv_order;
        assign(cv_order, count_along(stimuli));
        sort_by(cv_order, [&](size_type i) { return stimuli_cv[i]; });

        std::size_t n = stimuli.size();
        config.cv.reserve(n);
        config.cv_unique.reserve(n);
        config.frequency.reserve(n);
        config.phase.reserve(n);
        config.envelope_time.reserve(n);
        config.envelope_amplitude.reserve(n);

        for (auto i: cv_order) {
            const i_clamp& stim = stimuli[i].item;
            auto cv = stimuli_cv[i];
            double cv_area_scale = 1000./D.cv_area[cv]; // constant scale from nA/µm² to A/m².

            config.cv.push_back(cv);
            config.frequency.push_back(stim.frequency);
            config.phase.push_back(stim.phase);

            std::size_t envl_n = stim.envelope.size();
            std::vector<double> envl_t, envl_a;
            envl_t.reserve(envl_n);
            envl_a.reserve(envl_n);

            for (auto [t, a]: stim.envelope) {
                envl_t.push_back(t);
                envl_a.push_back(a*cv_area_scale);
            }
            config.envelope_time.push_back(std::move(envl_t));
            config.envelope_amplitude.push_back(std::move(envl_a));
        }

        std::unique_copy(config.cv.begin(), config.cv.end(), std::back_inserter(config.cv_unique));
        config.cv_unique.shrink_to_fit();

        if (!config.cv.empty()) M.stimuli = std::move(config);
    }

    // Ions:
    auto initial_iconc_map = assignments.get<init_int_concentration>();
    auto initial_econc_map = assignments.get<init_ext_concentration>();
    auto initial_rvpot_map = assignments.get<init_reversal_potential>();

    for (const auto& [ion, cvs]: ion_support) {
        fvm_ion_config config;
        config.cv = cvs;

        auto n_cv = config.cv.size();
        config.init_iconc.resize(n_cv);
        config.init_econc.resize(n_cv);
        config.init_revpot.resize(n_cv);
        config.reset_iconc.resize(n_cv);
        config.reset_econc.resize(n_cv);

        auto global_ion_data = value_by_key(global_dflt.ion_data, ion).value();
        auto dflt_iconc = global_ion_data.init_int_concentration.value();
        auto dflt_econc = global_ion_data.init_ext_concentration.value();
        auto dflt_rvpot = global_ion_data.init_reversal_potential.value();

        if (auto ion_data = value_by_key(dflt.ion_data, ion)) {
            auto val = ion_data.value();
            dflt_iconc = val.init_int_concentration.value_or(dflt_iconc);
            dflt_econc = val.init_ext_concentration.value_or(dflt_econc);
            dflt_rvpot = val.init_reversal_potential.value_or(dflt_rvpot);
        }

        const auto& iconc_on_cable = initial_iconc_map[ion];
        const auto& econc_on_cable = initial_econc_map[ion];
        const auto& rvpot_on_cable = initial_rvpot_map[ion];

        auto pw_times = [](const pw_elements<double>& pwa, const pw_elements<double>& pwb) {
            return pw_zip_with(pwa, pwb, [](std::pair<double, double>, double a, double b) { return a*b; });
        };

        for (auto i: count_along(config.cv)) {
            auto cv = config.cv[i];
            if (D.cv_area[cv]==0) continue;

            const auto& cv_cables = D.geometry.cables(cv);
            for (const mcable c: cv_cables) {
                auto iconc = pw_over_cable(iconc_on_cable, c, dflt_iconc);
                auto econc = pw_over_cable(econc_on_cable, c, dflt_econc);
                auto rvpot = pw_over_cable(rvpot_on_cable, c, dflt_rvpot);

                config.reset_iconc[i] += embedding.integrate_area(c.branch, iconc);
                config.reset_econc[i] += embedding.integrate_area(c.branch, econc);
                config.init_revpot[i] += embedding.integrate_area(c.branch, rvpot);

                auto iconc_masked = pw_times(pw_over_cable(init_iconc_mask[ion], c, 1.), iconc);
                auto econc_masked = pw_times(pw_over_cable(init_econc_mask[ion], c, 1.), econc);

                config.init_iconc[i] += embedding.integrate_area(c.branch, iconc_masked);
                config.init_econc[i] += embedding.integrate_area(c.branch, econc_masked);
            }

            // Scale all by area
            double oo_cv_area = 1./D.cv_area[cv];
            config.reset_iconc[i] *= oo_cv_area;
            config.reset_econc[i] *= oo_cv_area;
            config.init_revpot[i] *= oo_cv_area;
            config.init_iconc[i]  *= oo_cv_area;
            config.init_econc[i]  *= oo_cv_area;
        }

        if (auto di = D.diffusive_ions.find(ion); di != D.diffusive_ions.end()) {
            config.is_diffusive = true;
            config.face_diffusivity = di->second.face_diffusivity;
        }

        config.econc_written = write_xo.count(ion);
        config.iconc_written = write_xi.count(ion);
        if (!config.cv.empty()) M.ions[ion] = std::move(config);
    }

    std::unordered_map<std::string, mechanism_desc> revpot_tbl;
    std::unordered_set<std::string> revpot_specified;

    for (const auto& ion: util::keys(gprop.ion_species)) {
        if (auto maybe_revpot = value_by_key(dflt.reversal_potential_method, ion)
                              | value_by_key(global_dflt.reversal_potential_method, ion))
        {
            const mechanism_desc& revpot = *maybe_revpot;
            mechanism_info info = catalogue[revpot.name()];
            if (info.kind != arb_mechanism_kind_reversal_potential) {
                throw cable_cell_error("expected reversal potential mechanism for ion " +ion +", got "+ revpot.name() +" which has " +arb_mechanism_kind_str(info.kind));
            }

            verify_mechanism(gprop, D, info, revpot);
            revpot_specified.insert(ion);

            bool writes_this_revpot = false;
            for (auto& [name, info]: info.ions) {
                if (info.write_reversal_potential) {
                    if (revpot_tbl.count(name)) {
                        auto& existing_revpot_desc = revpot_tbl.at(name);
                        if (existing_revpot_desc.name() != revpot.name() || existing_revpot_desc.values() != revpot.values()) {
                            throw cable_cell_error("inconsistent revpot ion assignment for mechanism "+revpot.name());
                        }
                    }
                    else {
                        revpot_tbl[name] = revpot;
                    }

                    writes_this_revpot |= name==ion;
                }
            }

            if (!writes_this_revpot) {
                throw cable_cell_error("revpot mechanism for ion "+ion+" does not write this reversal potential");
            }

            M.ions[ion].revpot_written = true;

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
                    config.kind = arb_mechanism_kind_reversal_potential;
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

                    if (!config.cv.empty()) M.mechanisms[revpot.name()] = std::move(config);
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

    M.target_divs = {0u, M.n_target};
    return M;
}

std::unordered_map<std::string, fvm_mechanism_config>
make_voltage_mechanism_config(const cable_cell_global_properties& gprop,
                              const region_assignment<voltage_process> assignments,
                              const mechanism_catalogue& catalogue,
                              iexpr_ptr unit_scale,
                              unsigned cell_idx,
                              const fvm_cv_discretization& D,
                              const concrete_embedding& embedding,
                              const mprovider& provider) {
    std::unordered_map<std::string, fvm_mechanism_config> result;
    std::unordered_set<mcable> voltage_support;
    for (const auto& [name, cables]: assignments) {
        auto info = catalogue[name];
        if (info.kind != arb_mechanism_kind_voltage) {
            throw cable_cell_error("expected voltage mechanism, got " +name +" which has " +arb_mechanism_kind_str(info.kind));
        }

        fvm_mechanism_config config;
        config.kind = arb_mechanism_kind_voltage;

        std::vector<std::string> param_names;
        assign(param_names, util::keys(info.parameters));
        sort(param_names);

        std::vector<double> param_dflt;
        std::size_t n_param = param_names.size();
        param_dflt.reserve(n_param);
        config.param_values.reserve(n_param);

        for (std::size_t i = 0; i<n_param; ++i) {
            const auto& p = param_names[i];
            config.param_values.emplace_back(p, std::vector<arb_value_type>{});
            param_dflt.push_back(info.parameters.at(p).default_value);
        }

        mcable_map<double> support;
        std::vector<mcable_map<std::pair<double, iexpr_ptr>>> param_maps;

        param_maps.resize(n_param);

        for (const auto& [cable, density_iexpr]: cables) {
            const auto& mech = density_iexpr.mech;

            verify_mechanism(gprop, D, info, mech);
            const auto& set_params = mech.values();

            support.insert(cable, 1.);
            for (std::size_t i = 0; i<n_param; ++i) {
                double value = value_by_key(set_params, param_names[i]).value_or(param_dflt[i]);
                param_maps[i].insert(cable, {value, unit_scale});
            }
        }

        std::vector<double> param_on_cv(n_param);

        for (auto cv: D.geometry.cell_cvs(cell_idx)) {
            double area = 0;
            util::fill(param_on_cv, 0.);

            for (mcable c: D.geometry.cables(cv)) {
                double area_on_cable = embedding.integrate_area(c.branch, pw_over_cable(support, c, 0.));
                if (!area_on_cable) continue;
                area += area_on_cable;
                for (std::size_t i = 0; i<n_param; ++i) {
                    auto map = param_maps[i];
                    auto fun = [&](const auto& c, const auto& x) {
                        return x.first * x.second->eval(provider, c);
                    };
                    auto pw = pw_over_cable(map, c, 0., fun);
                    param_on_cv[i] += embedding.integrate_area(c.branch, pw);
                }
            }

            if (area>0) {
                config.cv.push_back(cv);
                config.norm_area.push_back(area/D.cv_area[cv]);

                double oo_area = 1./area;
                for (auto i: count_along(param_on_cv)) {
                    config.param_values[i].second.push_back(param_on_cv[i]*oo_area);
                }
            }
        }

        for (const auto& [cable, _]: support) {
            if (voltage_support.count(cable)) {
                throw cable_cell_error("Multiple voltage processes on a single cable");
            }
            voltage_support.insert(cable);
        }
        if (!config.cv.empty()) result.emplace(name, std::move(config));
    }
    return result;
}


} // namespace arb
