#include <arborio/label_parse.hpp>
#include "arbor/morph/morphology.hpp"
#include "common_cells.hpp"

namespace arb {
using namespace arborio::literals;

// Generate a segment tree from a sequence of points and parent index.
arb::segment_tree segments_from_points(
        std::vector<arb::mpoint> points,
        std::vector<arb::msize_t> parents,
        std::vector<int> tags)
{
    using arb::mnpos;
    using arb::msize_t;

    const auto np = points.size();

    // Sanity check the input.
    if (parents.size()!=np || np<2) {
        throw std::runtime_error("segments_from_points: every point must have a parent.");
    }
    if (tags.size() && tags.size()!=np) {
        throw std::runtime_error("segments_from_points: every point must have a tag.");
    }
    auto tag = [&tags](msize_t i) {return tags.size()? tags[i]: 1;};
    arb::segment_tree tree;
    std::unordered_map<msize_t, msize_t> segmap;

    tree.append(mnpos, points[0], points[1], tag(1));
    segmap[0] = mnpos;
    segmap[1] = 0;
    for (unsigned i=2; i<np; ++i) {
        auto p = segmap[parents[i]];
        if (p==mnpos) {
            tree.append(p, points[0], points[i], tag(i));
        }
        else {
            tree.append(p, points[i], tag(i));
        }
        segmap[i] = i-1;
    }

    return tree;
}


int soma_cell_builder::get_tag(const std::string& name) {
    auto it = tag_map.find(name);
    // If the name is not in the map, make a unique tag.
    // Tags start from 1.
    if (it==tag_map.end()) {
        tag_map[name] = ++tag_count;
        return tag_count;
    }
    return it->second;
}

soma_cell_builder::soma_cell_builder(double r) {
    auto tag = get_tag("soma");
    tree.append(arb::mnpos, {0,0,0,r}, {0,0,2*r,r}, tag);
    cv_boundaries.push_back({0, 1.});
    branch_distal_id.push_back(0);
    n_soma_children = 0;
}

mlocation soma_cell_builder::location(mlocation loc) const {
    if (loc.branch>=branch_distal_id.size()) {
        throw cable_cell_error("location not on cable_cell.");
    }
    if (n_soma_children==0) {
        return loc;
    }
    if (n_soma_children==1) {
        if (loc.branch<2) {
            double soma_len  = tree.segments()[branch_distal_id[0]].dist.z;
            double total_len = tree.segments()[branch_distal_id[1]].dist.z;
            // relative position of the end of the soma on the first branch
            double split = soma_len/total_len;
            double pos = loc.branch==0?
                split*loc.pos:
                split + (1-split)*loc.pos;
            return {0, pos};
        }
        return {loc.branch-1, loc.pos};
    }
    return loc;
}

mcable soma_cell_builder::cable(mcable cab) const {
    if (cab.branch>=branch_distal_id.size()) {
        throw cable_cell_error("cable not on cable_cell.");
    }
    auto beg = location({cab.branch, cab.prox_pos});
    auto end = location({cab.branch, cab.dist_pos});
    return {beg.branch, beg.pos, end.pos};
}

// Add a new branch that is attached to parent_branch.
// Returns the id of the new branch.
msize_t soma_cell_builder::add_branch(
        msize_t parent_branch,
        double len, double r1, double r2, int ncomp,
        const std::string& region)
{
    // Get tag id of region (add a new tag if region does not already exist).
    int tag = get_tag(region);

    msize_t p = branch_distal_id[parent_branch];
    auto& ploc =  tree.segments()[p].dist;

    double z = ploc.z;
    if (ploc.radius!=r1) {
        p = tree.append(p, {0,0,z,r1}, tag);
    }
    if (ncomp>1) {
        double dz = len/ncomp;
        double dr = (r2-r1)/ncomp;
        for (auto i=1; i<ncomp; ++i) {
            p = tree.append(p, {0,0,z+i*dz, r1+i*dr}, tag);
        }
    }
    p = tree.append(p, {0,0,z+len,r2}, tag);
    branch_distal_id.push_back(p);

    msize_t bid = branch_distal_id.size()-1;
    for (int i = 0; i<ncomp; ++i) {
        cv_boundaries.push_back(mlocation{bid, (2*i+1.)/(2.*ncomp)});
    }

    if (!parent_branch) ++n_soma_children;

    return bid;
}

cable_cell_description soma_cell_builder::make_cell() const {
    // Test that a valid tree was generated, that is, every branch has
    // either 0 children, or at least 2 children.
    for (auto i: branch_distal_id) {
        // skip soma
        if (i<2) continue;
        if (!tree.is_fork(i) && !tree.is_terminal(i)) {
            throw cable_cell_error(
                "attempt to construct a cable_cell from a soma_cell_builder "
                "where a non soma branch has only one child branch.");
        }
    }

    // Make label dictionary with one entry for each tag.
    label_dict dict;
    for (auto& tag: tag_map) {
        dict.set(tag.first, reg::tagged(tag.second));
    }

    auto boundaries = cv_boundaries;
    for (auto& b: boundaries) {
        b = location(b);
    }
    decor decorations;
    decorations.set_default(cv_policy_explicit(boundaries));
    // Construct cable_cell from sample tree, dictionary and decorations.
    return {std::move(tree), std::move(dict), std::move(decorations)};
}

/*
 * Create cell with just a soma:
 *
 * Soma:
 *    diameter: 18.8 µm
 *    mechanisms: HH (default params)
 *    bulk resistivitiy: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Stimuli:
 *    soma centre, t=[10 ms, 110 ms), 0.1 nA
 */

cable_cell_description make_cell_soma_only(bool with_stim) {
    soma_cell_builder builder(18.8/2.0);

    auto c = builder.make_cell();
    c.decorations.paint("soma"_lab, density("hh"));
    if (with_stim) {
        c.decorations.place(builder.location({0,0.5}), i_clamp{10., 100., 0.1}, "cc");
    }

    return {c.morph, c.labels, c.decorations};
}

/*
 * Create cell with a soma and unbranched dendrite:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    diameter: 1 µm
 *    length: 200 µm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

cable_cell_description make_cell_ball_and_stick(bool with_stim) {
    soma_cell_builder builder(12.6157/2.0);
    builder.add_branch(0, 200, 1.0/2, 1.0/2, 4, "dend");

    auto c = builder.make_cell();
    c.decorations.paint("soma"_lab, density("hh"));
    c.decorations.paint("dend"_lab, density("pas"));
    if (with_stim) {
        c.decorations.place(builder.location({1,1}), i_clamp{5, 80, 0.3}, "cc");
    }

    return {c.morph, c.labels, c.decorations};
}

/*
 * Create cell with a soma and three-branch dendrite with single branch point:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrites:
 *    mechanisms: passive (default params)
 *    diameter: 1 µm
 *    length: 100 µm
 *    compartments: 4
 *
 * Stimulus:
 *    end of first terminal branch, t=[5 ms, 85 ms), 0.45 nA
 *    end of second terminal branch, t=[40 ms, 50 ms), -0.2 nA
 */

cable_cell_description make_cell_ball_and_3stick(bool with_stim) {
    soma_cell_builder builder(12.6157/2.0);
    builder.add_branch(0, 100, 0.5, 0.5, 4, "dend");
    builder.add_branch(1, 100, 0.5, 0.5, 4, "dend");
    builder.add_branch(1, 100, 0.5, 0.5, 4, "dend");

    auto c = builder.make_cell();
    c.decorations.paint("soma"_lab, density("hh"));
    c.decorations.paint("dend"_lab, density("pas"));
    if (with_stim) {
        c.decorations.place(builder.location({2,1}), i_clamp{5.,  80., 0.45}, "cc0");
        c.decorations.place(builder.location({3,1}), i_clamp{40., 10.,-0.2}, "cc1");
    }

    return {c.morph, c.labels, c.decorations};
}

} // namespace arb
