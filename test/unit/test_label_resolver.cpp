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

        auto resolver = label_resolver(cell_labels_and_gids({sizes, labels, ranges}, gids));

        // Check resolver map correctness
        // gid 0
        auto map_0 = resolver.mapper.at(0);
        EXPECT_EQ(1u, map_0.at("l0_0").first.ranges.size());
        EXPECT_EQ(lid_range(0, 1), map_0.at("l0_0").first.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), map_0.at("l0_0").first.ranges_partition);

        // gid 1
        EXPECT_TRUE(resolver.mapper.at(1).empty());

        // gid 2
        auto map_2 = resolver.mapper.at(2);
        EXPECT_EQ(1u, map_2.at("l2_0").first.ranges.size());
        EXPECT_EQ(lid_range(0, 3), map_2.at("l2_0").first.ranges.front());
        EXPECT_EQ((vec{0u, 3u}), map_2.at("l2_0").first.ranges_partition);

        // gid 3
        auto map_3 = resolver.mapper.at(3);
        EXPECT_EQ(1u, map_3.at("l3_0").first.ranges.size());
        EXPECT_EQ(lid_range(1, 2), map_3.at("l3_0").first.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), map_3.at("l3_0").first.ranges_partition);

        EXPECT_EQ(1u, map_3.at("l3_1").first.ranges.size());
        EXPECT_EQ(lid_range(4, 10), map_3.at("l3_1").first.ranges.front());
        EXPECT_EQ((vec{0u, 6u}), map_3.at("l3_1").first.ranges_partition);

        // gid 4
        auto map_4 = resolver.mapper.at(4);
        EXPECT_EQ(1u, map_4.at("l4_0").first.ranges.size());
        EXPECT_EQ(lid_range(5, 6), map_4.at("l4_0").first.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), map_4.at("l4_0").first.ranges_partition);

        auto matches = map_4.at("l4_1").first.ranges;
        EXPECT_EQ(2u, matches.size());
        EXPECT_EQ(lid_range(8, 11), matches.at(0));
        EXPECT_EQ(lid_range(12, 14), matches.at(1));
        EXPECT_EQ((vec{0u, 3u, 5u}), map_4.at("l4_1").first.ranges_partition);

        // Check lid resolution
        // Non-existent gid or label
        EXPECT_THROW(resolver.get_lid({9, "l9_0"}), arb::bad_connection_label);
        EXPECT_THROW(resolver.get_lid({0, "l1_0"}), arb::bad_connection_label);
        EXPECT_THROW(resolver.get_lid({1, "l1_0"}), arb::bad_connection_label);

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

        EXPECT_EQ(8u,  resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(9u,  resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(10u, resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(12u, resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(13u, resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(8u,  resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(9u,  resolver.get_lid({4, "l4_1", lid_selection_policy::round_robin}));

        EXPECT_EQ(4u, resolver.get_lid({3, "l3_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(5u, resolver.get_lid({3, "l3_1", lid_selection_policy::round_robin}));

        // Reset and default policy
        resolver.reset();
        EXPECT_EQ(4u, resolver.get_lid({3, "l3_1"}));
        EXPECT_EQ(5u, resolver.get_lid({3, "l3_1"}));
        EXPECT_EQ(6u, resolver.get_lid({3, "l3_1"}));
        EXPECT_EQ(7u, resolver.get_lid({3, "l3_1"}));

        // Test Exception
        gids.push_back(5);
        sizes.push_back(1);
        labels.push_back("l5_0");
        ranges.push_back({0, 0});
        EXPECT_THROW(label_resolver(cell_labels_and_gids({sizes, labels, ranges}, gids)), arb::bad_connection_set);

        ranges.back() = {4, 2};
        EXPECT_THROW(label_resolver(cell_labels_and_gids({sizes, labels, ranges}, gids)), arb::bad_connection_range);

    }
    // multivalent labels
    {
        std::vector<cell_gid_type> gids = {0, 1, 2};
        std::vector<cell_size_type> sizes = {3, 0, 6};
        std::vector<cell_tag_type> labels = {"l0_0", "l0_1", "l0_0", "l2_0", "l2_0", "l2_2", "l2_1", "l2_1", "l2_2"};
        std::vector<lid_range> ranges = {{0, 1}, {0, 3}, {1, 3}, {4, 6}, {9, 12}, {5, 5}, {1, 2}, {0, 1}, {22, 23}};

        auto resolver = label_resolver(cell_labels_and_gids({sizes, labels, ranges}, gids));

        // Check resolver map correctness
        // gid 0
        auto map_0 = resolver.mapper.at(0);
        EXPECT_EQ(1u, map_0.count("l0_1"));
        EXPECT_EQ(lid_range(0, 3), map_0.at("l0_1").first.ranges.front());
        EXPECT_EQ((vec{0u, 3u}), map_0.at("l0_1").first.ranges_partition);

        auto matches = map_0.at("l0_0").first.ranges;
        EXPECT_EQ(2u, matches.size());
        EXPECT_EQ(lid_range(0, 1), matches.at(0));
        EXPECT_EQ(lid_range(1, 3), matches.at(1));
        EXPECT_EQ((vec{0u, 1u, 3u}), map_0.at("l0_0").first.ranges_partition);

        // gid 1
        EXPECT_TRUE(resolver.mapper.at(1).empty());

        // gid 2
        auto map_2 = resolver.mapper.at(2);
        matches = map_2.at("l2_0").first.ranges;
        EXPECT_EQ(2u, matches.size());
        EXPECT_EQ(lid_range(4, 6), matches.at(0));
        EXPECT_EQ(lid_range(9, 12), matches.at(1));
        EXPECT_EQ((vec{0u, 2u, 5u}), map_2.at("l2_0").first.ranges_partition);

        matches = map_2.at("l2_1").first.ranges;
        EXPECT_EQ(2u, matches.size());
        EXPECT_EQ(lid_range(1, 2), matches.at(0));
        EXPECT_EQ(lid_range(0, 1), matches.at(1));
        EXPECT_EQ((vec{0u, 1u, 2u}), map_2.at("l2_1").first.ranges_partition);

        matches = map_2.at("l2_2").first.ranges;
        EXPECT_EQ(2u, matches.size());
        EXPECT_EQ(lid_range(5, 5), matches.at(0));
        EXPECT_EQ(lid_range(22, 23), matches.at(1));
        EXPECT_EQ((vec{0u, 0u, 1u}), map_2.at("l2_2").first.ranges_partition);


        // Check lid resolution

        // gid 0
        EXPECT_EQ(0u, resolver.get_lid({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(1u, resolver.get_lid({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(2u, resolver.get_lid({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, resolver.get_lid({0, "l0_1", lid_selection_policy::round_robin}));

        EXPECT_THROW(resolver.get_lid({0, "l0_0", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
        EXPECT_EQ(0u, resolver.get_lid({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(1u, resolver.get_lid({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(2u, resolver.get_lid({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, resolver.get_lid({0, "l0_0", lid_selection_policy::round_robin}));

        // gid 2
        EXPECT_THROW(resolver.get_lid({2, "l2_0", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
        EXPECT_EQ(4u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(5u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(9u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(10u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(11u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(4u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(5u, resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));

        EXPECT_THROW(resolver.get_lid({2, "l2_1", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
        EXPECT_EQ(1u, resolver.get_lid({2, "l2_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, resolver.get_lid({2, "l2_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(1u, resolver.get_lid({2, "l2_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, resolver.get_lid({2, "l2_1", lid_selection_policy::round_robin}));

        EXPECT_EQ(22u, resolver.get_lid({2, "l2_2", lid_selection_policy::assert_univalent}));
        EXPECT_EQ(22u, resolver.get_lid({2, "l2_2", lid_selection_policy::round_robin}));
        EXPECT_EQ(22u, resolver.get_lid({2, "l2_2", lid_selection_policy::round_robin}));
        EXPECT_EQ(22u, resolver.get_lid({2, "l2_2", lid_selection_policy::assert_univalent}));

    }
}

