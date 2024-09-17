#include <arbor/morph/morphology.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborio/debug.hpp>

#include <gtest/gtest.h>

TEST(debug_io, single) {
    arb::segment_tree tree;
    arb::msize_t par = arb::mnpos;
    tree.append(par, {0, 0, 0, 5}, {0, 0, 10, 5}, 42);

    EXPECT_EQ("[-- id=0 --]\n", arborio::show(tree));
    EXPECT_EQ("<-- id=0 len=1 -->\n", arborio::show(arb::morphology{tree}));
}

TEST(debug_io, fork) {
    arb::segment_tree tree;
    arb::msize_t par = arb::mnpos;
    par = tree.append(par, {0, 0, 0, 5}, {0, 0, 10, 5}, 42);
    tree.append(par, {0, 0, 10, 5}, {0,  1, 10, 5}, 23);
    tree.append(par, {0, 0, 10, 5}, {0, -1, 10, 5}, 23);

    EXPECT_EQ("[-- id=0 --]-+-[-- id=1 --]\n"
              "             +-[-- id=2 --]\n",
              arborio::show(tree));
    EXPECT_EQ("<-- id=0 len=1 -->-+-<-- id=1 len=1 -->\n"
              "                   +-<-- id=2 len=1 -->\n",
              arborio::show(arb::morphology{tree}));
}

TEST(debug_io, complex) {
    arb::segment_tree tree;
    arb::msize_t lvl0 = arb::mnpos;
    lvl0 = tree.append(lvl0, {0, 0, 0, 5}, {0, 0, 10, 5}, 42);
    tree.append(lvl0, {0, 0, 10, 5}, {0,  1, 10, 5}, 23);
    auto lvl1 = tree.append(lvl0, {0, 0, 10, 5}, {0, -1, 10, 5}, 23);
    tree.append(lvl1, {0, -1, 10, 5}, { 1, -1, 10, 5}, 23);
    tree.append(lvl1, {0, -1, 10, 5}, {-1, -1, 10, 5}, 23);

    EXPECT_EQ("[-- id=0 --]-+-[-- id=1 --]\n"
              "             +-[-- id=2 --]-+-[-- id=3 --]\n"
              "                            +-[-- id=4 --]\n",
              arborio::show(tree));
    EXPECT_EQ("<-- id=0 len=1 -->-+-<-- id=1 len=1 -->\n"
              "                   +-<-- id=2 len=1 -->-+-<-- id=3 len=1 -->\n"
              "                                        +-<-- id=4 len=1 -->\n",
              arborio::show(arb::morphology{tree}));
}
