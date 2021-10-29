#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/morph/cv_geometry.hpp>
#include <arbor/morph/embed_pwlin.hpp>

#include "util/partition.hpp"
#include "util/piecewise.hpp"
#include "util/pw_over_cable.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "util/unique.hpp"

namespace arb {

auto cell_cv_geometry::cables(fvm_size_type cv_index) const {
    auto partn = util::partition_view(cv_cables_divs);
    return util::subrange_view(cv_cables, partn[cv_index]);
}

auto cell_cv_geometry::children(fvm_size_type cv_index) const {
    auto partn = util::partition_view(cv_children_divs);
    return util::subrange_view(cv_children, partn[cv_index]);
}

auto cell_cv_geometry::parent(fvm_size_type cv_index) const {
    return cv_parent[cv_index];
}

fvm_size_type cell_cv_geometry::num_cv() const {
    return cv_parent.size();
}

cell_cv_geometry cv_geometry_from_locset(const cable_cell& cell, const locset& lset) {
    auto pop = [](auto& vec) { auto h = vec.back(); return vec.pop_back(), h; };

    cell_cv_geometry geom;
    const auto& mp = cell.provider();
    const auto& m = mp.morphology();

    if (m.empty()) {
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

        util::sort(cables);
        util::append(geom.cv_cables, std::move(cables));
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
    util::stable_sort_by(geom.cv_children, [&geom](auto cv) { return geom.cv_parent[cv]; });

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

    // Build location query map.

    for (auto cv: util::make_span(n_cv)) {
        for (auto cable: geom.cables(cv)) {
            if (cable.branch>=geom.branch_cv_map.size()) {
                geom.branch_cv_map.resize(cable.branch+1);
            }

            // Ordering of CV ensures CV cables on any given branch are found sequentially.
            geom.branch_cv_map[cable.branch].push_back(cable.prox_pos, cable.dist_pos, cv);
        }
    }

    return geom;
}

auto region_cv_geometry::cables(fvm_size_type cv_index) const {
    auto partn = util::partition_view(cv_cables_divs);
    return util::subrange_view(cv_cables, partn[cv_index]);
}

auto region_cv_geometry::proportion(fvm_size_type cv_index) const {
    return cv_proportion[cv_index];
}

fvm_size_type region_cv_geometry::num_cv() const {
    return cv_proportion.size();
}

region_cv_geometry intersect_region(const cable_cell& cell, const region& reg, const cell_cv_geometry& geom) {
    const auto& mp = cell.provider();
    const auto& embedding = cell.embedding();

    region_cv_geometry reg_geom;
    auto extent = thingify(reg, mp);
    mcable_map<double> support;
    for (auto& cable: extent) {
        support.insert(cable, 1.);
    }
    if(support.empty()) {
        return reg_geom;
    }

    reg_geom.cv_cables_divs.push_back(0);
    for (auto cv: util::make_span(geom.num_cv())) {
        std::vector<mcable> cables;
        double cv_area = 0, area_on_cv = 0;
        for (mcable c: geom.cables(cv)) {
            cv_area += embedding.integrate_area(c);
            auto area_on_cable = embedding.integrate_area(c.branch, util::pw_over_cable(support, c, 0.));
            if (area_on_cable > 0) {
                cables.push_back(c);
                area_on_cv += area_on_cable;
            }
        }
        if (!cables.empty()) {
            util::append(reg_geom.cv_cables, cables);
            reg_geom.cv_cables_divs.push_back(reg_geom.cv_cables.size());
            reg_geom.cv_proportion.push_back(area_on_cv/cv_area);
        }
    }
    return reg_geom;
}

} //namespace arb