#include "../gtest.h"

#include <vector>

#include <arbor/arbexcept.hpp>

#include "label_resolver.hpp"
#include "util/rangeutil.hpp"

using namespace arb;

TEST(label_resolver, policies) {
    using vec = std::vector<cell_lid_type>;
    {
        std::vector<cell_gid_type> gids = {0, 1, 2, 3, 4};
        std::vector<cell_size_type> sizes = {1, 0, 1, 2, 3};
        std::vector<cell_tag_type> labels = {"l0_0", "l2_0", "l3_0", "l3_1", "l4_0", "l4_1", "l4_1"};
        std::vector<lid_range> ranges = {{0, 1}, {0, 3}, {1, 2}, {4, 10}, {5, 6}, {8, 11}, {12, 14}};

        auto res_map = label_resolution_map(cell_labels_and_gids({sizes, labels, ranges}, gids));

        // Check resolver map correctness
        // gid 0
        EXPECT_TRUE(res_map.find(0, "l0_0"));
        auto rset = res_map.at(0, "l0_0");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(0, 1), rset.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), rset.ranges_partition);

        // gid 1
        EXPECT_FALSE(res_map.find(1, "l0_0"));

        // gid 2
        EXPECT_TRUE(res_map.find(2, "l2_0"));
        rset = res_map.at(2, "l2_0");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(0, 3), rset.ranges.front());
        EXPECT_EQ((vec{0u, 3u}), rset.ranges_partition);

        // gid 3
        EXPECT_TRUE(res_map.find(3, "l3_0"));
        rset = res_map.at(3, "l3_0");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(1, 2), rset.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), rset.ranges_partition);

        EXPECT_TRUE(res_map.find(3, "l3_1"));
        rset = res_map.at(3, "l3_1");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(4, 10), rset.ranges.front());
        EXPECT_EQ((vec{0u, 6u}), rset.ranges_partition);

        // gid 4
        EXPECT_TRUE(res_map.find(4, "l4_0"));
        rset = res_map.at(4, "l4_0");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(5, 6), rset.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), rset.ranges_partition);

        EXPECT_TRUE(res_map.find(4, "l4_1"));
        rset = res_map.at(4, "l4_1");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(8, 11), rset.ranges.at(0));
        EXPECT_EQ(lid_range(12, 14), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 3u, 5u}), rset.ranges_partition);

        // Check lid resolution
        auto lid_resolver = arb::resolver();
        // Non-existent gid or label
        EXPECT_THROW(lid_resolver.resolve({9, "l9_0"}, res_map), arb::bad_connection_label);
        EXPECT_THROW(lid_resolver.resolve({0, "l1_0"}, res_map), arb::bad_connection_label);
        EXPECT_THROW(lid_resolver.resolve({1, "l1_0"}, res_map), arb::bad_connection_label);

        // Univalent
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::assert_univalent}, res_map));
        EXPECT_EQ(5u, lid_resolver.resolve({4, "l4_0", lid_selection_policy::assert_univalent}, res_map));
        EXPECT_EQ(5u, lid_resolver.resolve({4, "l4_0", lid_selection_policy::assert_univalent}, res_map)); // Repeated request

        EXPECT_THROW(lid_resolver.resolve({2, "l2_0", lid_selection_policy::assert_univalent}, res_map), arb::bad_univalent_connection_label);
        EXPECT_THROW(lid_resolver.resolve({3, "l3_1", lid_selection_policy::assert_univalent}, res_map), arb::bad_univalent_connection_label);
        EXPECT_THROW(lid_resolver.resolve({4, "l4_1", lid_selection_policy::assert_univalent}, res_map), arb::bad_univalent_connection_label);

        // Round robin
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}, res_map));

        EXPECT_EQ(0u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(1u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(2u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(0u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));

        EXPECT_EQ(5u, lid_resolver.resolve({4, "l4_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(5u, lid_resolver.resolve({4, "l4_0", lid_selection_policy::round_robin}, res_map));

        EXPECT_EQ(8u,  lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(9u,  lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(10u, lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(12u, lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(13u, lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(8u,  lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(9u,  lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}, res_map));

        EXPECT_EQ(4u, lid_resolver.resolve({3, "l3_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(5u, lid_resolver.resolve({3, "l3_1", lid_selection_policy::round_robin}, res_map));

        // Reset and default policy
        lid_resolver = arb::resolver();
        EXPECT_EQ(4u, lid_resolver.resolve({3, "l3_1"}, res_map));
        EXPECT_EQ(5u, lid_resolver.resolve({3, "l3_1"}, res_map));
        EXPECT_EQ(6u, lid_resolver.resolve({3, "l3_1"}, res_map));
        EXPECT_EQ(7u, lid_resolver.resolve({3, "l3_1"}, res_map));

        // Test Exception
        gids.push_back(5);
        sizes.push_back(1);
        labels.emplace_back("l5_0");
        ranges.emplace_back(0, 0);
        EXPECT_THROW(label_resolution_map(cell_labels_and_gids({sizes, labels, ranges}, gids)), arb::bad_connection_set);

        ranges.back() = {4, 2};
        EXPECT_THROW(label_resolution_map(cell_labels_and_gids({sizes, labels, ranges}, gids)), arb::bad_connection_range);

    }
    // multivalent labels
    {
        std::vector<cell_gid_type> gids = {0, 1, 2};
        std::vector<cell_size_type> sizes = {3, 0, 6};
        std::vector<cell_tag_type> labels = {"l0_0", "l0_1", "l0_0", "l2_0", "l2_0", "l2_2", "l2_1", "l2_1", "l2_2"};
        std::vector<lid_range> ranges = {{0, 1}, {0, 3}, {1, 3}, {4, 6}, {9, 12}, {5, 5}, {1, 2}, {0, 1}, {22, 23}};

        auto res_map = label_resolution_map(cell_labels_and_gids({sizes, labels, ranges}, gids));

        // Check resolver map correctness
        // gid 0
        EXPECT_TRUE(res_map.find(0, "l0_1"));
        auto rset = res_map.at(0, "l0_1");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(0, 3), rset.ranges.front());
        EXPECT_EQ((vec{0u, 3u}), rset.ranges_partition);

        EXPECT_TRUE(res_map.find(0, "l0_0"));
        rset = res_map.at(0, "l0_0");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(0, 1), rset.ranges.at(0));
        EXPECT_EQ(lid_range(1, 3), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 1u, 3u}), rset.ranges_partition);

        // gid 1
        EXPECT_FALSE(res_map.find(1, "l0_1"));

        // gid 2
        EXPECT_TRUE(res_map.find(2, "l2_0"));
        rset = res_map.at(2, "l2_0");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(4, 6), rset.ranges.at(0));
        EXPECT_EQ(lid_range(9, 12), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 2u, 5u}), rset.ranges_partition);

        EXPECT_TRUE(res_map.find(2, "l2_1"));
        rset = res_map.at(2, "l2_1");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(1, 2), rset.ranges.at(0));
        EXPECT_EQ(lid_range(0, 1), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 1u, 2u}), rset.ranges_partition);

        EXPECT_TRUE(res_map.find(2, "l2_2"));
        rset = res_map.at(2, "l2_2");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(5, 5), rset.ranges.at(0));
        EXPECT_EQ(lid_range(22, 23), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 0u, 1u}), rset.ranges_partition);

        // Check lid resolution
        auto lid_resolver = arb::resolver();
        // gid 0
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(1u, lid_resolver.resolve({0, "l0_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(2u, lid_resolver.resolve({0, "l0_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_1", lid_selection_policy::round_robin}, res_map));

        EXPECT_THROW(lid_resolver.resolve({0, "l0_0", lid_selection_policy::assert_univalent}, res_map), arb::bad_univalent_connection_label);
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(1u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(2u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}, res_map));

        // gid 2
        EXPECT_THROW(lid_resolver.resolve({2, "l2_0", lid_selection_policy::assert_univalent}, res_map), arb::bad_univalent_connection_label);
        EXPECT_EQ(4u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(5u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(9u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(10u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(11u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(4u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(5u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}, res_map));

        EXPECT_THROW(lid_resolver.resolve({2, "l2_1", lid_selection_policy::assert_univalent}, res_map), arb::bad_univalent_connection_label);
        EXPECT_EQ(1u, lid_resolver.resolve({2, "l2_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(0u, lid_resolver.resolve({2, "l2_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(1u, lid_resolver.resolve({2, "l2_1", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(0u, lid_resolver.resolve({2, "l2_1", lid_selection_policy::round_robin}, res_map));

        EXPECT_EQ(22u, lid_resolver.resolve({2, "l2_2", lid_selection_policy::assert_univalent}, res_map));
        EXPECT_EQ(22u, lid_resolver.resolve({2, "l2_2", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(22u, lid_resolver.resolve({2, "l2_2", lid_selection_policy::round_robin}, res_map));
        EXPECT_EQ(22u, lid_resolver.resolve({2, "l2_2", lid_selection_policy::assert_univalent}, res_map));

    }
}

