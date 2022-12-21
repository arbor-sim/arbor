#include <iostream>
#include <fstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborio/neurolucida.hpp>

#include <gtest/gtest.h>
#include "arbor/arbexcept.hpp"

TEST(asc, file_not_found) {
    EXPECT_THROW(arborio::load_asc("this-file-does-not-exist.asc"), arb::file_not_found_error);
}

// Declare the implementation of the parser that takes a string input
namespace arborio {
asc_morphology parse_asc_string(const char* input);
}

// Test different forms of empty files.
TEST(asc, empty_file) {
    // A file with no contents at all.
    {
        const char* empty_file = "";
        auto m = arborio::parse_asc_string(empty_file);
        EXPECT_TRUE(m.morphology.empty());
    }

    // Inputs with header meta-data, but no sub-trees.
    {
        const char * input =
" (Description \
)  ;  End of description \
(ImageCoords Filename \
 \"C:\\local\\input.lsm\" \
 Merge 65535 65535 65535 0 \
 Coords 0.057299 0.057299 0 0 0 0.057299 0.057299 0 0 -0.360149 0.057299 \
 0.057299 0 0 -0.720299 0.057299 0.057299 0 0 -1.080448 0.057299 0.057299 0 0 \
 -1.440598 0.057299 0.057299 0 0 -1.800747 0.057299 0.057299 0 0 -2.160896 \
) ; End of ImageCoord";
        auto m = arborio::parse_asc_string(input);
        EXPECT_TRUE(m.morphology.empty());
    }
}

// Morphologies with only the CellBody
TEST(asc, only_cell_body) {
    { // CellBody with no samples: should be an error.
        const char* input = "((CellBody))";
        EXPECT_THROW(arborio::parse_asc_string(input), arborio::asc_parse_error);
    }

    { // CellBody with a single sample defining the center and radius of a sphere.
        const char* input = "((CellBody) (0 0 1 2))";
        auto m = arborio::parse_asc_string(input);

        // Soma is a cylinder composed of two cylinders attached to the root.
        EXPECT_EQ(m.morphology.num_branches(), 2u);

        auto& segs1 = m.morphology.branch_segments(0);
        EXPECT_EQ(segs1.size(), 1u);
        EXPECT_EQ(segs1[0].prox, (arb::mpoint{0., 0., 1., 1.}));
        EXPECT_EQ(segs1[0].dist, (arb::mpoint{0.,-1., 1., 1.}));

        auto& segs2 = m.morphology.branch_segments(1);
        EXPECT_EQ(segs2.size(), 1u);
        EXPECT_EQ(segs2[0].prox, (arb::mpoint{0., 0., 1., 1.}));
        EXPECT_EQ(segs2[0].dist, (arb::mpoint{0., 1., 1., 1.}));
    }


    { // CellBody with a circular contour defined by 4 points
        const char* input = "((CellBody) (-2 0 1 0) (0 2 1 0) (2 0 1 0) (0 -2 1 0))";
        auto m = arborio::parse_asc_string(input);

        // Soma is a cylinder composed of two cylinders attached to the root.
        EXPECT_EQ(m.morphology.num_branches(), 2u);

        auto& segs1 = m.morphology.branch_segments(0);
        EXPECT_EQ(segs1.size(), 1u);
        EXPECT_EQ(segs1[0].prox, (arb::mpoint{0., 0., 1., 2.}));
        EXPECT_EQ(segs1[0].dist, (arb::mpoint{0.,-2., 1., 2.}));

        auto& segs2 = m.morphology.branch_segments(1);
        EXPECT_EQ(segs2.size(), 1u);
        EXPECT_EQ(segs2[0].prox, (arb::mpoint{0., 0., 1., 2.}));
        EXPECT_EQ(segs2[0].dist, (arb::mpoint{0., 2., 1., 2.}));
    }

    { // Cell with two CellBodys: unsupported feature ATM.
        const char* input =
"((CellBody)\
  (-2 0 1 0)\
  (0 2 1 0))\
 ((CellBody)\
  (-2 0 3 0)\
  (0 2 3 0))";
        EXPECT_THROW(arborio::parse_asc_string(input), arborio::asc_unsupported);
    }
}

// Test parsing of basic meta data that can be added to the start of a 
// This information is discarded and not used for building the morphology,
// however it is still required that it should be parsed, and throw an error
// if ill-formed or unexpected meta-data is encountered.
TEST(asc, sub_tree_meta) {
    {   // String 
        const char * input = "(\"Soma\" (CellBody) (237.86 -189.71 -6.49 0.06))";
        auto m = arborio::parse_asc_string(input);
        EXPECT_EQ(m.morphology.num_branches(), 2u);
    }

    {  // Named color
        const char * input = "((Color Red) (CellBody) (237.86 -189.71 -6.49 0.06))";
        auto m = arborio::parse_asc_string(input);
        EXPECT_EQ(m.morphology.num_branches(), 2u);
    }

    {   // RGB color
        const char * input = "((Color RGB(128, 128, 96)) (CellBody) (237.86 -189.71 -6.49 0.06))";
        auto m = arborio::parse_asc_string(input);
        EXPECT_EQ(m.morphology.num_branches(), 2u);
    }

    {   // badly formatted RGB color: missing comma
        const char * input = "((Color RGB(128  128, 96)) (CellBody) (237.86 -189.71 -6.49 0.06))";
        EXPECT_THROW(arborio::parse_asc_string(input), arborio::asc_parse_error);
    }

    {   // badly formatted RGB color: out of range triple value
        const char * input = "((Color RGB(128,  128, 256)) (CellBody) (237.86 -189.71 -6.49 0.06))";
        EXPECT_THROW(arborio::parse_asc_string(input), arborio::asc_parse_error);
    }
}

// Soma composed of 2 branches, and stick and fork dendrite composed of 3 branches.
const char *asc_ball_and_y_dendrite =
"((CellBody)\
 (0 0 0 4)\
)\
((Dendrite)\
 (0 2 0 2)\
 (0 5 0 2)\
 (\
  (-5 5 0 2)\
  |\
  (6 5 0 2)\
 )\
 )";

// Soma composed of 2 branches, and a dendrite with a bit more interesting branching.
const char *asc_ball_and_fancy_dendrite=
"((CellBody)\
 (0 0 0 4)\
)\
((Dendrite)\
 (0 2 0 2)\
 (0 5 0 2)\
 (\
  (-5 5 0 2)\
  (\
   (-5 5 0 2)\
   |\
   (6 5 0 2)\
  )\
  |\
  (6 5 0 2)\
 )\
 )";

// Soma composed of 2 branches, and stick and fork dendrite and axon
// composed of 3 branches each.
const char* asc_ball_and_y_dendrite_and_y_axon =
"((CellBody)\
 (0 0 0 4)\
)\
((Dendrite)\
 (0 2 0 2)\
 (0 5 0 2)\
 (\
  (-5 5 0 2)\
  |\
  (6 5 0 2)\
 )\
)\
((Axon)\
 (0 -2 0 2)\
 (0 -5 0 2)\
 (\
  (-5 -5 0 2)\
  |\
  (6 -5 0 2)\
 )\
)";

TEST(asc, branching) {
    {
        auto result = arborio::parse_asc_string(asc_ball_and_y_dendrite);
        const auto& m = result.morphology;
        EXPECT_EQ(m.num_branches(), 5u);
        EXPECT_EQ(m.branch_children(0).size(), 0u);
        EXPECT_EQ(m.branch_children(1).size(), 0u);
        EXPECT_EQ(m.branch_children(2).size(), 2u);
        EXPECT_EQ(m.branch_children(2)[0], 3u);
        EXPECT_EQ(m.branch_children(2)[1], 4u);
        EXPECT_EQ(m.branch_children(3).size(), 0u);
        EXPECT_EQ(m.branch_children(4).size(), 0u);
    }
    {
        auto result = arborio::parse_asc_string(asc_ball_and_fancy_dendrite);
        const auto& m = result.morphology;
        EXPECT_EQ(m.num_branches(), 7u);
        EXPECT_EQ(m.branch_children(0).size(), 0u);
        EXPECT_EQ(m.branch_children(1).size(), 0u);
        EXPECT_EQ(m.branch_children(2).size(), 2u);
        EXPECT_EQ(m.branch_children(2)[0], 3u);
        EXPECT_EQ(m.branch_children(2)[1], 6u);
        EXPECT_EQ(m.branch_children(3).size(), 2u);
        EXPECT_EQ(m.branch_children(3)[0], 4u);
        EXPECT_EQ(m.branch_children(3)[1], 5u);
        EXPECT_EQ(m.branch_children(4).size(), 0u);
        EXPECT_EQ(m.branch_children(5).size(), 0u);
        EXPECT_EQ(m.branch_children(6).size(), 0u);
    }
    {
        auto result = arborio::parse_asc_string(asc_ball_and_y_dendrite_and_y_axon);
        const auto& m = result.morphology;
        EXPECT_EQ(m.num_branches(), 8u);
        EXPECT_EQ(m.branch_children(0).size(), 0u);
        EXPECT_EQ(m.branch_children(1).size(), 0u);
        EXPECT_EQ(m.branch_children(2).size(), 2u);
        EXPECT_EQ(m.branch_children(2)[0], 3u);
        EXPECT_EQ(m.branch_children(2)[1], 4u);
        EXPECT_EQ(m.branch_children(3).size(), 0u);
        EXPECT_EQ(m.branch_children(4).size(), 0u);
        EXPECT_EQ(m.branch_children(5).size(), 2u);
        EXPECT_EQ(m.branch_children(5)[0], 6u);
        EXPECT_EQ(m.branch_children(5)[1], 7u);
        EXPECT_EQ(m.branch_children(6).size(), 0u);
        EXPECT_EQ(m.branch_children(7).size(), 0u);
    }
}

// Test that
//  * first segment in branches connected to the soma has as its
//    proximal sample the first sample in the branch
//  * all other non-soma branches have the distal end of the parent
//    branch as their proximal sample.
//  * the soma is composed of two branches, attached to the soma center,
//    and extending along the z axis
TEST(asc, soma_connection) {
    {
        auto result = arborio::parse_asc_string(asc_ball_and_y_dendrite);
        const auto& m = result.morphology;
        EXPECT_EQ(m.num_branches(), 5u);
        // Test soma
        EXPECT_EQ(m.branch_segments(0)[0].prox, (arb::mpoint{0, 0, 0, 2}));
        EXPECT_EQ(m.branch_segments(0)[0].dist, (arb::mpoint{0,-2, 0, 2}));
        EXPECT_EQ(m.branch_segments(1)[0].prox, (arb::mpoint{0, 0, 0, 2}));
        EXPECT_EQ(m.branch_segments(1)[0].dist, (arb::mpoint{0, 2, 0, 2}));
        // Test dendrite proximal ends
        EXPECT_EQ(m.branch_segments(2)[0].prox, (arb::mpoint{0, 2, 0, 1}));
        EXPECT_EQ(m.branch_segments(3)[0].prox, (arb::mpoint{0, 5, 0, 1}));
        EXPECT_EQ(m.branch_segments(4)[0].prox, (arb::mpoint{0, 5, 0, 1}));
    }
    {
        auto result = arborio::parse_asc_string(asc_ball_and_fancy_dendrite);
        const auto& m = result.morphology;
        EXPECT_EQ(m.num_branches(), 7u);
        // Test soma
        EXPECT_EQ(m.branch_segments(0)[0].prox, (arb::mpoint{0, 0, 0, 2}));
        EXPECT_EQ(m.branch_segments(0)[0].dist, (arb::mpoint{0,-2, 0, 2}));
        EXPECT_EQ(m.branch_segments(1)[0].prox, (arb::mpoint{0, 0, 0, 2}));
        EXPECT_EQ(m.branch_segments(1)[0].dist, (arb::mpoint{0, 2, 0, 2}));
        // Test dendrite proximal ends
        EXPECT_EQ(m.branch_segments(2)[0].prox, (arb::mpoint{ 0, 2, 0, 1}));
        EXPECT_EQ(m.branch_segments(3)[0].prox, (arb::mpoint{ 0, 5, 0, 1}));
        EXPECT_EQ(m.branch_segments(4)[0].prox, (arb::mpoint{-5, 5, 0, 1}));
        EXPECT_EQ(m.branch_segments(5)[0].prox, (arb::mpoint{-5, 5, 0, 1}));
        EXPECT_EQ(m.branch_segments(6)[0].prox, (arb::mpoint{ 0, 5, 0, 1}));
    }
    {
        auto result = arborio::parse_asc_string(asc_ball_and_y_dendrite_and_y_axon);
        const auto& m = result.morphology;
        EXPECT_EQ(m.num_branches(), 8u);
        // Test soma
        EXPECT_EQ(m.branch_segments(0)[0].prox, (arb::mpoint{0, 0, 0, 2}));
        EXPECT_EQ(m.branch_segments(0)[0].dist, (arb::mpoint{0,-2, 0, 2}));
        EXPECT_EQ(m.branch_segments(1)[0].prox, (arb::mpoint{0, 0, 0, 2}));
        EXPECT_EQ(m.branch_segments(1)[0].dist, (arb::mpoint{0, 2, 0, 2}));
        // Test dendrite proximal ends
        EXPECT_EQ(m.branch_segments(2)[0].prox, (arb::mpoint{0, 2, 0, 1}));
        EXPECT_EQ(m.branch_segments(3)[0].prox, (arb::mpoint{0, 5, 0, 1}));
        EXPECT_EQ(m.branch_segments(4)[0].prox, (arb::mpoint{0, 5, 0, 1}));
        // Test axon proximal ends
        EXPECT_EQ(m.branch_segments(5)[0].prox, (arb::mpoint{0,-2, 0, 1}));
        EXPECT_EQ(m.branch_segments(6)[0].prox, (arb::mpoint{0,-5, 0, 1}));
        EXPECT_EQ(m.branch_segments(7)[0].prox, (arb::mpoint{0,-5, 0, 1}));
    }
}
