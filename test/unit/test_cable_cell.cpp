#include <gtest/gtest.h>
#include "../common_cells.hpp"

#include <arbor/cable_cell.hpp>
#include <arbor/cable_cell_param.hpp>

#include <arborio/label_parse.hpp>

using namespace arb;
using namespace arborio::literals;

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
    dict.set("term", "(terminal)"_ls);

    decor decorations;

    mlocation_list empty_sites = {};
    mlocation_list three_sites = {{0, 0.1}, {1, 0.2}, {1, 0.7}};

    // Place synapses and threshold detectors in interleaved order.
    // Note: there are 2 terminal points.
    decorations.place("term"_lab, synapse("expsyn"), "t0");
    decorations.place("term"_lab, synapse("expsyn"), "t1");
    decorations.place("term"_lab, threshold_detector{-10}, "s0");
    decorations.place(empty_sites, synapse("expsyn"), "t2");
    decorations.place("term"_lab, threshold_detector{-20}, "s1");
    decorations.place(three_sites, synapse("expsyn"), "t3");
    decorations.place("term"_lab, synapse("exp2syn"), "t3");

    cable_cell cell(morph, decorations, dict);

    // Get the assigned lid ranges for each placement
    const auto& src_ranges = cell.detector_ranges();
    const auto& tgt_ranges = cell.synapse_ranges();

    EXPECT_EQ(1u, tgt_ranges.count("t0"));
    EXPECT_EQ(1u, tgt_ranges.count("t1"));
    EXPECT_EQ(1u, src_ranges.count("s0"));
    EXPECT_EQ(1u, tgt_ranges.count("t2"));
    EXPECT_EQ(1u, src_ranges.count("s1"));
    EXPECT_EQ(2u, tgt_ranges.count("t3"));

    auto r1 = tgt_ranges.equal_range("t0").first->second;
    auto r2 = tgt_ranges.equal_range("t1").first->second;
    auto r3 = src_ranges.equal_range("s0").first->second;
    auto r4 = tgt_ranges.equal_range("t2").first->second;
    auto r5 = src_ranges.equal_range("s1").first->second;

    auto r6_range = tgt_ranges.equal_range("t3");
    auto r6_0 = r6_range.first;
    auto r6_1 = std::next(r6_range.first);
    if (r6_0->second.begin != 4u) {
        std::swap(r6_0, r6_1);
    }

    EXPECT_EQ(r1.begin, 0u); EXPECT_EQ(r1.end, 2u);
    EXPECT_EQ(r2.begin, 2u); EXPECT_EQ(r2.end, 4u);
    EXPECT_EQ(r3.begin, 0u); EXPECT_EQ(r3.end, 2u);
    EXPECT_EQ(r4.begin, 4u); EXPECT_EQ(r4.end, 4u);
    EXPECT_EQ(r5.begin, 2u); EXPECT_EQ(r5.end, 4u);
    EXPECT_EQ(r6_0->second.begin, 4u); EXPECT_EQ(r6_0->second.end, 7u);
    EXPECT_EQ(r6_1->second.begin, 7u); EXPECT_EQ(r6_1->second.end, 9u);
}
