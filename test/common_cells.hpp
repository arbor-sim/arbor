#include <cmath>

#include <arbor/cable_cell.hpp>
#include <arbor/mechinfo.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/recipe.hpp>

namespace arb {

class soma_cell_builder {
    double soma_rad;
    sample_tree tree;
    std::vector<msize_t> branch_distal_id;
    std::unordered_map<std::string, int> tag_map;
    locset cv_boundaries = mlocation{0, 1.};
    int tag_count = 0;

    // Get tag id of region.
    // Add a new tag if region with that name has not already had a tag associated with it.
    int get_tag(const std::string& name) {
        auto it = tag_map.find(name);
        // If the name is not in the map, make a unique tag.
        // Tags start from 1.
        if (it==tag_map.end()) {
            tag_map[name] = ++tag_count;
            return tag_count;
        }
        return it->second;
    }

public:
    soma_cell_builder(double r): soma_rad(r) {
        tree.append({{0,0,0,r}, get_tag("soma")});
        branch_distal_id.push_back(0);
    }

    // Add a new branch that is attached to parent_branch.
    // Returns the id of the new branch.
    msize_t add_branch(msize_t parent_branch, double len, double r1, double r2, int ncomp,
                    const std::string& region)
    {
        // Get tag id of region (add a new tag if region does not already exist).
        int tag = get_tag(region);

        msize_t p = branch_distal_id[parent_branch];
        double z = parent_branch? tree.samples()[p].loc.z: soma_rad;

        p = tree.append(p, {{0,0,z,r1}, tag});
        if (ncomp>1) {
            double dz = len/ncomp;
            double dr = (r2-r1)/ncomp;
            for (auto i=1; i<ncomp; ++i) {
                p = tree.append(p, {{0,0,z+i*dz, r1+i*dr}, tag});
            }
        }
        p = tree.append(p, {{0,0,z+len,r2}, tag});
        branch_distal_id.push_back(p);

        msize_t bid = branch_distal_id.size()-1;
        for (int i = 0; i<ncomp; ++i) {
            cv_boundaries = sum(cv_boundaries,  mlocation{bid, (2*i+1.)/(2.*ncomp)});
        }
        return bid;
    }

    cable_cell make_cell() const {
        // Test that a valid tree was generated, that is, every branch has
        // either 0 children, or at least 2 children.
        for (auto i: branch_distal_id) {
            if (i==0) continue;
            auto prop = tree.properties()[i];
            if (!is_fork(prop) && !is_terminal(prop)) {
                throw cable_cell_error(
                    "attempt to construct a cable_cell from a soma_cell_builder "
                    "where a branch has only one child branch.");
            }
        }

        // Make label dictionary with one entry for each tag.
        label_dict dict;
        for (auto& tag: tag_map) {
            dict.set(tag.first, reg::tagged(tag.second));
        }

        // Make cable_cell from sample tree and dictionary.
        cable_cell c(tree, dict);
        c.default_parameters.discretization = cv_policy_explicit(cv_boundaries);
        return c;
    }
};

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

inline cable_cell make_cell_soma_only(bool with_stim = true) {
    soma_cell_builder builder(18.8/2.0);

    auto c = builder.make_cell();
    c.paint("soma", "hh");
    if (with_stim) {
        c.place(mlocation{0,0.5}, i_clamp{10., 100., 0.1});
    }

    return c;
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

inline cable_cell make_cell_ball_and_stick(bool with_stim = true) {
    soma_cell_builder builder(12.6157/2.0);
    builder.add_branch(0, 200, 1.0/2, 1.0/2, 4, "dend");

    auto c = builder.make_cell();
    c.paint("soma", "hh");
    c.paint("dend", "pas");
    if (with_stim) {
        c.place(mlocation{1,1}, i_clamp{5, 80, 0.3});
    }

    return c;
}

/*
 * Create cell with a soma and three-branch dendrite with single branch point:
 *
 * O----======
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

inline cable_cell make_cell_ball_and_3stick(bool with_stim = true) {
    soma_cell_builder builder(12.6157/2.0);
    builder.add_branch(0, 100, 0.5, 0.5, 4, "dend");
    builder.add_branch(1, 100, 0.5, 0.5, 4, "dend");
    builder.add_branch(1, 100, 0.5, 0.5, 4, "dend");

    auto c = builder.make_cell();
    c.paint("soma", "hh");
    c.paint("dend", "pas");
    if (with_stim) {
        c.place(mlocation{2,1}, i_clamp{5.,  80., 0.45});
        c.place(mlocation{3,1}, i_clamp{40., 10.,-0.2});
    }

    return c;
}

} // namespace arb
