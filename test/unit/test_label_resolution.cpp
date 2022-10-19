#include <gtest/gtest.h>

#include <vector>

#include <arbor/arbexcept.hpp>

#include "label_resolution.hpp"

using namespace arb;

TEST(test_cell_label_range, build) {
    using ivec = std::vector<cell_size_type>;
    using svec = std::vector<cell_tag_type>;
    using lvec = std::vector<lid_range>;

    // Test add_cell and add_label
    auto b0 = cell_label_range();
    EXPECT_THROW(b0.add_label("l0", {0u, 1u}), arb::arbor_internal_error);
    EXPECT_TRUE(b0.sizes().empty());
    EXPECT_TRUE(b0.labels().empty());
    EXPECT_TRUE(b0.ranges().empty());
    EXPECT_TRUE(b0.check_invariant());

    auto b1 = cell_label_range();
    b1.add_cell();
    b1.add_cell();
    b1.add_cell();
    EXPECT_EQ((ivec{0u, 0u, 0u}), b1.sizes());
    EXPECT_TRUE(b1.labels().empty());
    EXPECT_TRUE(b1.ranges().empty());
    EXPECT_TRUE(b1.check_invariant());

    auto b2 = cell_label_range();
    b2.add_cell();
    b2.add_label("l0", {0u, 1u});
    b2.add_label("l0", {3u, 13u});
    b2.add_label("l1", {0u, 5u});
    b2.add_cell();
    b2.add_cell();
    b2.add_label("l2", {6u, 8u});
    b2.add_label("l3", {1u, 0u});
    b2.add_label("l4", {7u, 2u});
    b2.add_label("l4", {7u, 2u});
    b2.add_label("l2", {7u, 2u});
    EXPECT_EQ((ivec{3u, 0u, 5u}), b2.sizes());
    EXPECT_EQ((svec{"l0", "l0", "l1", "l2", "l3", "l4", "l4", "l2"}), b2.labels());
    EXPECT_EQ((lvec{{0u, 1u}, {3u, 13u}, {0u, 5u}, {6u, 8u}, {1u, 0u}, {7u, 2u}, {7u, 2u}, {7u, 2u}}), b2.ranges());
    EXPECT_TRUE(b2.check_invariant());

    auto b3 = cell_label_range();
    b3.add_cell();
    b3.add_label("r0", {0u, 9u});
    b3.add_label("r1", {10u, 10u});
    b3.add_cell();
    EXPECT_EQ((ivec{2u, 0u}), b3.sizes());
    EXPECT_EQ((svec{"r0", "r1"}), b3.labels());
    EXPECT_EQ((lvec{{0u, 9u}, {10u, 10u}}), b3.ranges());
    EXPECT_TRUE(b3.check_invariant());

    // Test appending
    b0.append(b1);
    EXPECT_EQ((ivec{0u, 0u, 0u}), b0.sizes());
    EXPECT_TRUE(b0.labels().empty());
    EXPECT_TRUE(b0.ranges().empty());
    EXPECT_TRUE(b0.check_invariant());

    b0.append(b2);
    EXPECT_EQ((ivec{0u, 0u, 0u, 3u, 0u, 5u}), b0.sizes());
    EXPECT_EQ((svec{"l0", "l0", "l1", "l2", "l3", "l4", "l4", "l2"}), b0.labels());
    EXPECT_EQ((lvec{{0u, 1u}, {3u, 13u}, {0u, 5u}, {6u, 8u}, {1u, 0u}, {7u, 2u}, {7u, 2u}, {7u, 2u}}), b0.ranges());
    EXPECT_TRUE(b0.check_invariant());

    b0.append(b3);
    EXPECT_EQ((ivec{0u, 0u, 0u, 3u, 0u, 5u, 2u, 0u}), b0.sizes());
    EXPECT_EQ((svec{"l0", "l0", "l1", "l2", "l3", "l4", "l4", "l2", "r0", "r1"}), b0.labels());
    EXPECT_EQ((lvec{{0u, 1u}, {3u, 13u}, {0u, 5u}, {6u, 8u}, {1u, 0u}, {7u, 2u}, {7u, 2u}, {7u, 2u}, {0u, 9u}, {10u, 10u}}), b0.ranges());
    EXPECT_TRUE(b0.check_invariant());
}

TEST(test_cell_labels_and_gids, build) {
    // Test add_cell and add_label
    auto b0 = cell_label_range();
    EXPECT_THROW(cell_labels_and_gids(b0, {1u}), arb::arbor_internal_error);
    auto c0 = cell_labels_and_gids(b0, {});

    auto b1 = cell_label_range();
    b1.add_cell();
    auto c1 = cell_labels_and_gids(b1, {0});
    c0.append(c1);
    EXPECT_TRUE(c0.check_invariant());

    auto b2 = cell_label_range();
    b2.add_cell();
    b2.add_cell();
    b2.add_cell();
    auto c2 = cell_labels_and_gids(b2, {4, 6, 0});
    c0.append(c2);
    EXPECT_TRUE(c0.check_invariant());

    auto b3 = cell_label_range();
    b3.add_cell();
    b3.add_cell();
    auto c3 = cell_labels_and_gids(b3, {1, 1});
    c0.append(c3);
    EXPECT_TRUE(c0.check_invariant());

    c0.gids = {};
    EXPECT_FALSE(c0.check_invariant());

    c0.gids = {0, 4, 6, 0, 1, 1};
    EXPECT_TRUE(c0.check_invariant());

    c0.label_range = {};
    EXPECT_FALSE(c0.check_invariant());
}

TEST(test_label_resolution, policies) {
    using vec = std::vector<cell_lid_type>;
    {
        std::vector<cell_gid_type> gids = {0, 1, 2, 3, 4};
        std::vector<cell_size_type> sizes = {1, 0, 1, 2, 3};
        std::vector<cell_tag_type> labels = {"l0_0", "l2_0", "l3_0", "l3_1", "l4_0", "l4_1", "l4_1"};
        std::vector<lid_range> ranges = {{0, 1}, {0, 3}, {1, 2}, {4, 10}, {5, 6}, {8, 11}, {12, 14}};

        auto res_map = label_resolution_map(cell_labels_and_gids({sizes, labels, ranges}, gids));

        // Check resolver map correctness
        // gid 0
        EXPECT_EQ(1u, res_map.count(0, "l0_0"));
        auto rset = res_map.at(0, "l0_0");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(0, 1), rset.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), rset.ranges_partition);

        // gid 1
        EXPECT_EQ(0u, res_map.count(1, "l0_0"));

        // gid 2
        EXPECT_EQ(1u, res_map.count(2, "l2_0"));
        rset = res_map.at(2, "l2_0");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(0, 3), rset.ranges.front());
        EXPECT_EQ((vec{0u, 3u}), rset.ranges_partition);

        // gid 3
        EXPECT_EQ(1u, res_map.count(3, "l3_0"));
        rset = res_map.at(3, "l3_0");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(1, 2), rset.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), rset.ranges_partition);

        EXPECT_EQ(1u, res_map.count(3, "l3_1"));
        rset = res_map.at(3, "l3_1");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(4, 10), rset.ranges.front());
        EXPECT_EQ((vec{0u, 6u}), rset.ranges_partition);

        // gid 4
        EXPECT_EQ(1u, res_map.count(4, "l4_0"));
        rset = res_map.at(4, "l4_0");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(5, 6), rset.ranges.front());
        EXPECT_EQ((vec{0u, 1u}), rset.ranges_partition);

        EXPECT_EQ(1u, res_map.count(4, "l4_1"));
        rset = res_map.at(4, "l4_1");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(8, 11), rset.ranges.at(0));
        EXPECT_EQ(lid_range(12, 14), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 3u, 5u}), rset.ranges_partition);

        // Check lid resolution
        auto lid_resolver = arb::resolver(&res_map);
        // Non-existent gid or label
        EXPECT_THROW(lid_resolver.resolve({9, "l9_0"}), arb::bad_connection_label);
        EXPECT_THROW(lid_resolver.resolve({0, "l1_0"}), arb::bad_connection_label);
        EXPECT_THROW(lid_resolver.resolve({1, "l1_0"}), arb::bad_connection_label);

        // Univalent
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::assert_univalent}));
        EXPECT_EQ(5u, lid_resolver.resolve({4, "l4_0", lid_selection_policy::assert_univalent}));
        EXPECT_EQ(5u, lid_resolver.resolve({4, "l4_0", lid_selection_policy::assert_univalent})); // Repeated request

        EXPECT_THROW(lid_resolver.resolve({2, "l2_0", lid_selection_policy::assert_univalent}), arb::bad_connection_label);
        EXPECT_THROW(lid_resolver.resolve({3, "l3_1", lid_selection_policy::assert_univalent}), arb::bad_connection_label);
        EXPECT_THROW(lid_resolver.resolve({4, "l4_1", lid_selection_policy::assert_univalent}), arb::bad_connection_label);

        // Round robin
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}));

        EXPECT_EQ(0u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(1u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(2u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));

        EXPECT_EQ(5u, lid_resolver.resolve({4, "l4_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(5u, lid_resolver.resolve({4, "l4_0", lid_selection_policy::round_robin}));

        EXPECT_EQ(8u,  lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(9u,  lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(10u, lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(12u, lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(13u, lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(8u,  lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(9u,  lid_resolver.resolve({4, "l4_1", lid_selection_policy::round_robin}));

        EXPECT_EQ(4u, lid_resolver.resolve({3, "l3_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(5u, lid_resolver.resolve({3, "l3_1", lid_selection_policy::round_robin}));

        // Reset
        lid_resolver = arb::resolver(&res_map);
        EXPECT_EQ(4u, lid_resolver.resolve({3, "l3_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(5u, lid_resolver.resolve({3, "l3_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(6u, lid_resolver.resolve({3, "l3_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(7u, lid_resolver.resolve({3, "l3_1", lid_selection_policy::round_robin}));

        // Test exception
        gids.push_back(5);
        sizes.push_back(1);
        labels.emplace_back("l5_0");
        ranges.emplace_back(0, 0);
        res_map = label_resolution_map(cell_labels_and_gids({sizes, labels, ranges}, gids));

        lid_resolver = arb::resolver(&res_map);
        EXPECT_THROW(lid_resolver.resolve({5, "l5_0"}), arb::bad_connection_label);

        ranges.back() = {4, 2};
        EXPECT_THROW(label_resolution_map(cell_labels_and_gids({sizes, labels, ranges}, gids)), arb::arbor_internal_error);

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
        EXPECT_EQ(1u, res_map.count(0, "l0_1"));
        auto rset = res_map.at(0, "l0_1");
        EXPECT_EQ(1u, rset.ranges.size());
        EXPECT_EQ(lid_range(0, 3), rset.ranges.front());
        EXPECT_EQ((vec{0u, 3u}), rset.ranges_partition);

        EXPECT_EQ(1u, res_map.count(0, "l0_0"));
        rset = res_map.at(0, "l0_0");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(0, 1), rset.ranges.at(0));
        EXPECT_EQ(lid_range(1, 3), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 1u, 3u}), rset.ranges_partition);

        // gid 1
        EXPECT_EQ(0u, res_map.count(1, "l0_1"));

        // gid 2
        EXPECT_EQ(1u, res_map.count(2, "l2_0"));
        rset = res_map.at(2, "l2_0");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(4, 6), rset.ranges.at(0));
        EXPECT_EQ(lid_range(9, 12), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 2u, 5u}), rset.ranges_partition);

        EXPECT_EQ(1u, res_map.count(2, "l2_1"));
        rset = res_map.at(2, "l2_1");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(1, 2), rset.ranges.at(0));
        EXPECT_EQ(lid_range(0, 1), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 1u, 2u}), rset.ranges_partition);

        EXPECT_EQ(1u, res_map.count(2, "l2_2"));
        rset = res_map.at(2, "l2_2");
        EXPECT_EQ(2u, rset.ranges.size());
        EXPECT_EQ(lid_range(5, 5), rset.ranges.at(0));
        EXPECT_EQ(lid_range(22, 23), rset.ranges.at(1));
        EXPECT_EQ((vec{0u, 0u, 1u}), rset.ranges_partition);

        // Check lid resolution
        auto lid_resolver = arb::resolver(&res_map);
        // gid 0
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(1u, lid_resolver.resolve({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(2u, lid_resolver.resolve({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_1", lid_selection_policy::round_robin}));

        EXPECT_THROW(lid_resolver.resolve({0, "l0_0", lid_selection_policy::assert_univalent}), arb::bad_connection_label);
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(1u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(2u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, lid_resolver.resolve({0, "l0_0", lid_selection_policy::round_robin}));

        // gid 2
        EXPECT_THROW(lid_resolver.resolve({2, "l2_0", lid_selection_policy::assert_univalent}), arb::bad_connection_label);
        EXPECT_EQ(4u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(5u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(9u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(10u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(11u, lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(4u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ(5u,  lid_resolver.resolve({2, "l2_0", lid_selection_policy::round_robin}));

        EXPECT_THROW(lid_resolver.resolve({2, "l2_1", lid_selection_policy::assert_univalent}), arb::bad_connection_label);
        EXPECT_EQ(1u, lid_resolver.resolve({2, "l2_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, lid_resolver.resolve({2, "l2_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(1u, lid_resolver.resolve({2, "l2_1", lid_selection_policy::round_robin}));
        EXPECT_EQ(0u, lid_resolver.resolve({2, "l2_1", lid_selection_policy::round_robin}));

        EXPECT_EQ(22u, lid_resolver.resolve({2, "l2_2", lid_selection_policy::assert_univalent}));
        EXPECT_EQ(22u, lid_resolver.resolve({2, "l2_2", lid_selection_policy::round_robin}));
        EXPECT_EQ(22u, lid_resolver.resolve({2, "l2_2", lid_selection_policy::round_robin}));
        EXPECT_EQ(22u, lid_resolver.resolve({2, "l2_2", lid_selection_policy::assert_univalent}));

    }
}

