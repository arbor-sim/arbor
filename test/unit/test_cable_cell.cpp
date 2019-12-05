#include "../gtest.h"

#include <arbor/cable_cell.hpp>
#include <arbor/math.hpp>
#include <arbor/morph/locset.hpp>

#include "io/sepval.hpp"

#include "tree.hpp"

using namespace arb;
using ::arb::math::pi;

TEST(cable_cell, soma) {
    // test that insertion of a soma works
    //      define with centre point @ (0,0,1)
    double soma_radius = 3.2;

    arb::sample_tree samples;
    samples.append({0,0,0,soma_radius,1});
    auto c = cable_cell(arb::morphology(samples));

    EXPECT_EQ(c.has_soma(), true);

    auto s = c.soma();
    EXPECT_EQ(s->radius(), soma_radius);
}

TEST(cable_cell, multiple_cables) {
    double soma_radius = std::pow(3./(4.*pi<double>), 1./3.);

    // Generate a cylindrical cable segment of length 1/pi and radius 1
    //      volume = 1
    //      area   = 2
    // Returns the distal point of the added cable.
    auto append_branch = [soma_radius](sample_tree& stree, int proximal) {
        constexpr int tag = 2;
        if (!proximal) {
            double z = soma_radius;
            proximal = stree.append(0, {0,0,z, 1/pi<double>, tag});
        }
        return stree.append(proximal, msample{0, 0, stree.samples()[proximal].loc.z+1, 1/pi<double>, tag});
    };

    // cell strucure with branch numbers
    //
    //          0
    //         / \.
    //        1   2
    //       / \.
    //      3   4

    arb::sample_tree samples;
    samples.append({0,0,-soma_radius,soma_radius,1});

    // hook the dendrite and axons
    append_branch(samples, 0);
    append_branch(samples, 0);
    append_branch(samples, 2);
    append_branch(samples, 2);

    auto c = cable_cell(arb::morphology(samples, true));

    EXPECT_EQ(c.num_branches(), 5u);

    // construct the graph
    tree con(c.parents());

    auto no_parent = tree::no_parent;
    EXPECT_EQ(con.num_segments(), 5u);
    EXPECT_EQ(con.parent(0), no_parent);
    EXPECT_EQ(con.parent(1), 0u);
    EXPECT_EQ(con.parent(2), 0u);
    EXPECT_EQ(con.parent(3), 1u);
    EXPECT_EQ(con.parent(4), 1u);
    EXPECT_EQ(con.num_children(0), 2u);
    EXPECT_EQ(con.num_children(1), 2u);
    EXPECT_EQ(con.num_children(2), 0u);
    EXPECT_EQ(con.num_children(3), 0u);
    EXPECT_EQ(con.num_children(4), 0u);
}
