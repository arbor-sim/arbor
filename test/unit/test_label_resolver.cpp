#include "../gtest.h"

#include <vector>

#include <arbor/arbexcept.hpp>

#include "label_resolver.hpp"
using namespace arb;

TEST(label_resolver, policies) {
    std::vector<cell_gid_type> gids = {0, 1, 2, 3, 4, 5};
    std::vector<cell_size_type> sizes = {1, 0, 1, 2, 2, 1};
    std::vector<cell_tag_type> labels = {"l0_0", "l2_0", "l3_0", "l3_1", "l4_0", "l4_1", "l5_0"};
    std::vector<lid_range> ranges = {{0, 1}, {0, 3}, {1, 0}, {4, 10}, {5, 6}, {8, 11}, {0, 0}};

    auto resolver = label_resolver(cell_labeled_ranges(gids, sizes, labels, ranges));

    unsigned label_idx = 0;
    for (unsigned i = 0; i < gids.size(); ++i) {
        auto gid = gids[i];
        auto size = sizes[i];
        EXPECT_TRUE(resolver.mapper.count(gid));
        for (unsigned j = 0; j < size; ++j) {
            EXPECT_TRUE(resolver.mapper.at(gid).count(labels[label_idx]));
            EXPECT_EQ(ranges[label_idx], resolver.mapper.at(gid).at(labels[label_idx]).first);
            label_idx++;
        }
    }

    // Non-existent gid or label
    EXPECT_THROW(resolver.get_lid({9, "l9_0"}), arb::bad_connection_label);
    EXPECT_THROW(resolver.get_lid({0, "l1_0"}), arb::bad_connection_label);
    EXPECT_THROW(resolver.get_lid({1, "l1_0"}), arb::bad_connection_label);

    // Invalid range
    EXPECT_THROW(resolver.get_lid({3, "l3_0"}), arb::bad_connection_range);
    EXPECT_THROW(resolver.get_lid({5, "l5_0"}), arb::bad_connection_range);

    // Univalent
    EXPECT_EQ(0u, resolver.get_lid({0, "l0_0", lid_selection_policy::assert_univalent}));
    EXPECT_EQ(5u, resolver.get_lid({4, "l4_0", lid_selection_policy::assert_univalent}));
    EXPECT_EQ(5u, resolver.get_lid({4, "l4_0", lid_selection_policy::assert_univalent})); // Repeated request

    EXPECT_THROW(resolver.get_lid({2, "l2_0", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
    EXPECT_THROW(resolver.get_lid({3, "l3_1", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
    EXPECT_THROW(resolver.get_lid({4, "l4_1", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);

    // Round robin
    EXPECT_EQ(0u, resolver.get_lid({0, "l0_0", lid_selection_policy::round_robin}));
    EXPECT_EQ(0u, resolver.get_lid({0, "l0_0", lid_selection_policy::round_robin}));

    EXPECT_EQ(0u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
    EXPECT_EQ(1u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
    EXPECT_EQ(2u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
    EXPECT_EQ(0u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));

    EXPECT_EQ(5u, resolver.get_lid({4, "l4_0", lid_selection_policy::round_robin}));
    EXPECT_EQ(5u, resolver.get_lid({4, "l4_0", lid_selection_policy::round_robin}));
    EXPECT_EQ(8u, resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
    EXPECT_EQ(9u, resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
    EXPECT_EQ(10u, resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
    EXPECT_EQ(8u, resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));

    EXPECT_EQ(4u, resolver.get_lid({3, "l3_1", lid_selection_policy::round_robin}));
    EXPECT_EQ(5u, resolver.get_lid({3, "l3_1", lid_selection_policy::round_robin}));

    // Reset and default policy
    resolver.reset();
    EXPECT_EQ(4u, resolver.get_lid({3, "l3_1"}));
    EXPECT_EQ(5u, resolver.get_lid({3, "l3_1"}));
    EXPECT_EQ(6u, resolver.get_lid({3, "l3_1"}));
    EXPECT_EQ(7u, resolver.get_lid({3, "l3_1"}));
}

