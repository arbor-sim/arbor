#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/morph/cv_data.hpp>
#include <arbor/morph/embed_pwlin.hpp>

#include "morph/cv_data.hpp"
#include "util/partition.hpp"
#include "util/pw_over_cable.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "util/unique.hpp"

namespace arb {

cell_cv_data_impl::cell_cv_data_impl(const cable_cell& cell, const locset& lset) {
    auto pop = [](auto& vec) { auto h = vec.back(); return vec.pop_back(), h; };

    const auto& mp = cell.provider();
    const auto& m = mp.morphology();

    if (m.empty()) {
        return;
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

    constexpr arb_index_type no_parent = -1;
    std::vector<std::pair<mlocation, arb_index_type>> next_cv_head; // head loc, parent cv index
    next_cv_head.emplace_back(mlocation{mnpos, 0}, no_parent);

    mcable_list cables;
    std::vector<msize_t> branches;
    cv_cables_divs.push_back(0);
    arb_index_type cv_index = 0;

    while (!next_cv_head.empty()) {
        auto next = pop(next_cv_head);
        mlocation h = next.first;

        cables.clear();
        branches.clear();
        branches.push_back(h.branch);
        cv_parent.push_back(next.second);

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
        util::append(cv_cables, std::move(cables));
        cv_cables_divs.push_back(cv_cables.size());
        ++cv_index;
    }

    auto n_cv = cv_index;
    arb_assert(n_cv>0);
    arb_assert(cv_parent.front()==-1);
    arb_assert(util::all_of(util::subrange_view(cv_parent, 1, n_cv),
                            [](auto v) { return v!=no_parent; }));

    // Construct CV children mapping by sorting CV indices by parent.
    assign(cv_children, util::make_span(1, n_cv));
    util::stable_sort_by(cv_children, [this](auto cv) { return cv_parent[cv]; });

    cv_children_divs.reserve(n_cv+1);
    cv_children_divs.push_back(0);

    auto b = cv_children.begin();
    auto e = cv_children.end();
    auto from = b;

    for (arb_index_type cv = 0; cv<n_cv; ++cv) {
        from = std::partition_point(from, e,
                                    [cv, this](auto i) { return cv_parent[i]<=cv; });
        cv_children_divs.push_back(from-b);
    }
}

mcable_list cell_cv_data::cables(arb_size_type cv_index) const {
    auto partn = util::partition_view(impl_->cv_cables_divs);
    auto view = util::subrange_view(impl_->cv_cables, partn[cv_index]);
    return mcable_list{view.begin(), view.end()};
}

std::vector<arb_index_type> cell_cv_data::children(arb_size_type cv_index) const {
    auto partn = util::partition_view(impl_->cv_children_divs);
    auto view = util::subrange_view(impl_->cv_children, partn[cv_index]);
    return std::vector<arb_index_type>{view.begin(), view.end()};
}

arb_index_type cell_cv_data::parent(arb_size_type cv_index) const {
    return impl_->cv_parent[cv_index];
}

arb_size_type cell_cv_data::size() const {
    return impl_->cv_parent.size();
}

ARB_ARBOR_API std::optional<cell_cv_data> cv_data(const cable_cell& cell) {
    if (auto policy = cell.decorations().defaults().discretization) {
        return cell_cv_data(cell, policy->cv_boundary_points(cell));
    }
    return {};
}

using impl_ptr = std::unique_ptr<cell_cv_data_impl, void (*)(cell_cv_data_impl*)>;
impl_ptr make_impl(cell_cv_data_impl* c) {
    return impl_ptr(c, [](cell_cv_data_impl* p){delete p;});
}

cell_cv_data::cell_cv_data(const cable_cell& cell, const locset& lset):
    impl_(make_impl(new cell_cv_data_impl(cell, lset))),
    provider_(cell.provider())
{}

ARB_ARBOR_API std::vector<cv_proportion> intersect_region(const region& reg, const cell_cv_data& geom, bool by_length) {
    const auto& mp = geom.provider();
    const auto& embedding = mp.embedding();

    std::vector<cv_proportion> intersection;
    auto extent = thingify(reg, mp);
    mcable_map<double> support;
    for (auto& cable: extent) {
        support.insert(cable, 1.);
    }
    if(support.empty()) {
        return {};
    }

    for (auto cv: util::make_span(geom.size())) {
        double cv_total = 0, reg_on_cv = 0;
        for (mcable c: geom.cables(cv)) {
            if (by_length) {
                cv_total += embedding.integrate_length(c);
                reg_on_cv += embedding.integrate_length(c.branch, util::pw_over_cable(support, c, 0.));
            }
            else {
                cv_total += embedding.integrate_area(c);
                reg_on_cv += embedding.integrate_area(c.branch, util::pw_over_cable(support, c, 0.));
            }
        }
        if (reg_on_cv>0) {
            intersection.push_back({cv, reg_on_cv/cv_total});
        }
    }
    return intersection;
}

} //namespace arb
