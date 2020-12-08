#include "../gtest.h"
#include "common_cells.hpp"

#include "s_expr.hpp"

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/string_literals.hpp>

using namespace arb;
using namespace arb::literals;

TEST(cable_cell, lid_ranges) {

    // Create a morphology with two branches:
    //   * branch 0 is a single segment soma followed by a single segment dendrite.
    //   * branch 1 is an axon attached to the root.
    segment_tree tree;
    tree.append(mnpos, {0, 0, 0, 10}, {0, 0, 10, 10}, 1);
    tree.append(0,     {0, 0, 10, 1}, {0, 0, 100, 1}, 3);
    tree.append(mnpos, {0, 0, 0, 2}, {0, 0, -20, 2}, 2);

    arb::morphology morph(tree);

    label_dict dict;
    dict.set("term", locset("(terminal)"));

    decor decorations;

    mlocation_list empty_sites = {};
    mlocation_list three_sites = {{0, 0.1}, {1, 0.2}, {1, 0.7}};

    // Place synapses and threshold detectors in interleaved order.
    // Note: there are 2 terminal points.
    auto idx1 = decorations.place("term"_lab, "expsyn");
    auto idx2 = decorations.place("term"_lab, "expsyn");
    auto idx3 = decorations.place("term"_lab, threshold_detector{-10});
    auto idx4 = decorations.place(empty_sites, "expsyn");
    auto idx5 = decorations.place("term"_lab, threshold_detector{-20});
    auto idx6 = decorations.place(three_sites, "expsyn");

    cable_cell cell(morph, dict, decorations);

    // Get the assigned lid ranges for each placement
    auto r1 = cell.placed_lid_range(idx1);
    auto r2 = cell.placed_lid_range(idx2);
    auto r3 = cell.placed_lid_range(idx3);
    auto r4 = cell.placed_lid_range(idx4);
    auto r5 = cell.placed_lid_range(idx5);
    auto r6 = cell.placed_lid_range(idx6);

    EXPECT_EQ(idx1, 0u); EXPECT_EQ(r1.begin, 0u); EXPECT_EQ(r1.end, 2u);
    EXPECT_EQ(idx2, 1u); EXPECT_EQ(r2.begin, 2u); EXPECT_EQ(r2.end, 4u);
    EXPECT_EQ(idx3, 2u); EXPECT_EQ(r3.begin, 0u); EXPECT_EQ(r3.end, 2u);
    EXPECT_EQ(idx4, 3u); EXPECT_EQ(r4.begin, 4u); EXPECT_EQ(r4.end, 4u);
    EXPECT_EQ(idx5, 4u); EXPECT_EQ(r5.begin, 2u); EXPECT_EQ(r5.end, 4u);
    EXPECT_EQ(idx6, 5u); EXPECT_EQ(r6.begin, 4u); EXPECT_EQ(r6.end, 7u);
}
