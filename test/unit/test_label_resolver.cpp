#include "../gtest.h"

#include <vector>

#include <arbor/arbexcept.hpp>

#include "label_resolver.hpp"
#include "util/rangeutil.hpp"

using namespace arb;

TEST(label_resolver, policies) {

    auto extract_sorted_range = [](const auto& match_range) {
      std::vector<lid_range> matches;
      for (auto it = match_range.first; it != match_range.second; ++it) {
          matches.push_back(it->second.first);
      }
      util::sort(matches);
      return matches;
    };

    {
        std::vector<cell_gid_type> gids = {0, 1, 2, 3, 4, 5};
        std::vector<cell_size_type> sizes = {1, 0, 1, 2, 3, 1};
        std::vector<cell_tag_type> labels = {"l0_0", "l2_0", "l3_0", "l3_1", "l4_0", "l4_1", "l4_1", "l5_0"};
        std::vector<lid_range> ranges = {{0, 1}, {0, 3}, {1, 0}, {4, 10}, {5, 6}, {8, 11}, {11, 13}, {0, 0}};

        auto resolver = label_resolver(cell_label_range(gids, sizes, labels, ranges));

        // Check resolver map correctness
        // gid 0
        auto map_0 = resolver.mapper.at(0);
        EXPECT_EQ(1u, map_0.count("l0_0"));
        EXPECT_EQ(lid_range(0, 1), map_0.equal_range("l0_0").first->second.first);

        // gid 1
        EXPECT_TRUE(resolver.mapper.at(1).empty());

        // gid 2
        auto map_2 = resolver.mapper.at(2);
        EXPECT_EQ(1u, map_2.count("l2_0"));
        EXPECT_EQ(lid_range(0, 3), map_2.equal_range("l2_0").first->second.first);

        // gid 3
        auto map_3 = resolver.mapper.at(3);
        EXPECT_EQ(1u, map_3.count("l3_0"));
        EXPECT_EQ(lid_range(1, 0), map_3.equal_range("l3_0").first->second.first);

        EXPECT_EQ(1u, map_3.count("l3_1"));
        EXPECT_EQ(lid_range(4, 10), map_3.equal_range("l3_1").first->second.first);

        // gid 4
        auto map_4 = resolver.mapper.at(4);
        EXPECT_EQ(1u, map_4.count("l4_0"));
        EXPECT_EQ(lid_range(5, 6), map_4.equal_range("l4_0").first->second.first);

        EXPECT_EQ(2u, map_4.count("l4_1"));
        auto matches = extract_sorted_range(map_4.equal_range("l4_1"));
        EXPECT_EQ(lid_range(8, 11), matches.at(0));
        EXPECT_EQ(lid_range(11, 13), matches.at(1));

        // gid 5
        auto map_5 = resolver.mapper.at(5);
        EXPECT_EQ(1u, map_5.count("l5_0"));
        EXPECT_EQ(lid_range(0, 0), map_5.equal_range("l5_0").first->second.first);

        // Check lid resolution
        using vec = std::vector<cell_lid_type>;

        auto resolve = [resolver](const cell_global_label_type& label) {
          auto lids = resolver.get_lid(label);
          util::sort(lids);
          return lids;
        };

        // Non-existent gid or label
        EXPECT_THROW(resolver.get_lid({9, "l9_0"}), arb::bad_connection_label);
        EXPECT_THROW(resolver.get_lid({0, "l1_0"}), arb::bad_connection_label);
        EXPECT_THROW(resolver.get_lid({1, "l1_0"}), arb::bad_connection_label);

        // Invalid range
        EXPECT_THROW(resolver.get_lid({3, "l3_0"}), arb::bad_connection_range);
        EXPECT_THROW(resolver.get_lid({5, "l5_0"}), arb::bad_connection_range);

        // Univalent
        EXPECT_EQ((vec{0u}), resolver.get_lid({0, "l0_0", lid_selection_policy::assert_univalent}));
        EXPECT_EQ((vec{5u}), resolver.get_lid({4, "l4_0", lid_selection_policy::assert_univalent}));
        EXPECT_EQ((vec{5u}), resolver.get_lid({4, "l4_0", lid_selection_policy::assert_univalent})); // Repeated request

        EXPECT_THROW(resolver.get_lid({2, "l2_0", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
        EXPECT_THROW(resolver.get_lid({3, "l3_1", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
        EXPECT_THROW(resolver.get_lid({4, "l4_1", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);

        // Round robin
        EXPECT_EQ((vec{0u}), resolver.get_lid({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{0u}), resolver.get_lid({0, "l0_0", lid_selection_policy::round_robin}));

        EXPECT_EQ((vec{0u}), resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{1u}), resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{2u}), resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{0u}), resolver.get_lid({2, "l2_0", lid_selection_policy::round_robin}));

        EXPECT_EQ((vec{5u}), resolver.get_lid({4, "l4_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{5u}), resolver.get_lid({4, "l4_0", lid_selection_policy::round_robin}));

        EXPECT_EQ((vec{8u, 11u}), resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{9u, 12u}), resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{10u, 11u}), resolve({4, "l4_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{8u, 12u}), resolve({4, "l4_1", lid_selection_policy::round_robin}));

        EXPECT_EQ((vec{4u}), resolver.get_lid({3, "l3_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{5u}), resolver.get_lid({3, "l3_1", lid_selection_policy::round_robin}));

        // Reset and default policy
        resolver.reset();
        EXPECT_EQ((vec{4u}), resolver.get_lid({3, "l3_1"}));
        EXPECT_EQ((vec{5u}), resolver.get_lid({3, "l3_1"}));
        EXPECT_EQ((vec{6u}), resolver.get_lid({3, "l3_1"}));
        EXPECT_EQ((vec{7u}), resolver.get_lid({3, "l3_1"}));
    }
    // multivalent labels
    {
        std::vector<cell_gid_type> gids = {0, 1, 2};
        std::vector<cell_size_type> sizes = {3, 0, 4};
        std::vector<cell_tag_type> labels = {"l0_0", "l0_1", "l0_0", "l2_0", "l2_0", "l2_1", "l2_1"};
        std::vector<lid_range> ranges = {{0, 1}, {0, 3}, {1, 3}, {4, 8}, {9, 12}, {0, 1}, {1, 2}};

        auto resolver = label_resolver(cell_label_range(gids, sizes, labels, ranges));

        // Check resolver map correctness
        // gid 0
        auto map_0 = resolver.mapper.at(0);
        EXPECT_EQ(1u, map_0.count("l0_1"));
        EXPECT_EQ(lid_range(0, 3), map_0.equal_range("l0_1").first->second.first);

        EXPECT_EQ(2u, map_0.count("l0_0"));
        auto matches = extract_sorted_range(map_0.equal_range("l0_0"));
        EXPECT_EQ(lid_range(0, 1), matches.at(0));
        EXPECT_EQ(lid_range(1, 3), matches.at(1));

        // gid 1
        EXPECT_TRUE(resolver.mapper.at(1).empty());

        // gid 2
        auto map_2 = resolver.mapper.at(2);
        EXPECT_EQ(2u, map_2.count("l2_0"));
        matches = extract_sorted_range(map_2.equal_range("l2_0"));
        EXPECT_EQ(lid_range(4, 8), matches.at(0));
        EXPECT_EQ(lid_range(9, 12), matches.at(1));

        EXPECT_EQ(2u, map_2.count("l2_1"));
        matches = extract_sorted_range(map_2.equal_range("l2_1"));
        EXPECT_EQ(lid_range(0, 1), matches.at(0));
        EXPECT_EQ(lid_range(1, 2), matches.at(1));

        // Check lid resolution
        using vec = std::vector<cell_lid_type>;

        auto resolve = [resolver](const cell_global_label_type& label) {
            auto lids = resolver.get_lid(label);
            util::sort(lids);
            return lids;
        };

        // gid 0
        EXPECT_EQ((vec{0u}), resolver.get_lid({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{1u}), resolver.get_lid({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{2u}), resolver.get_lid({0, "l0_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{0u}), resolver.get_lid({0, "l0_1", lid_selection_policy::round_robin}));

        EXPECT_THROW(resolver.get_lid({0, "l0_0", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
        EXPECT_EQ((vec{0u, 1u}), resolve({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{0u, 2u}), resolve({0, "l0_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{0u, 1u}), resolve({0, "l0_0", lid_selection_policy::round_robin}));

        // gid 2
        EXPECT_THROW(resolver.get_lid({2, "l2_0", lid_selection_policy::assert_univalent}), arb::bad_univalent_connection_label);
        EXPECT_EQ((vec{4u, 9u}), resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{5u, 10u}), resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{6u, 11u}), resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{7u, 9u}), resolve({2, "l2_0", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{4u, 10u}), resolve({2, "l2_0", lid_selection_policy::round_robin}));

        EXPECT_EQ((vec{0u, 1u}), resolve({2, "l2_1", lid_selection_policy::assert_univalent}));
        EXPECT_EQ((vec{0u, 1u}), resolve({2, "l2_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{0u, 1u}), resolve({2, "l2_1", lid_selection_policy::assert_univalent}));
        EXPECT_EQ((vec{0u, 1u}), resolve({2, "l2_1", lid_selection_policy::round_robin}));
        EXPECT_EQ((vec{0u, 1u}), resolve({2, "l2_1", lid_selection_policy::round_robin}));

    }
}

