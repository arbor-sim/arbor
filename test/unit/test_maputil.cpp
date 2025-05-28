#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <ranges>

#include <util/maputil.hpp>
#include <util/rangeutil.hpp>

#include "common.hpp"

using namespace arb;

using namespace std::string_literals;
using testing::nocopy;
using testing::nomove;

TEST(maputil, is_assoc) {
    using util::is_associative_container;

    EXPECT_TRUE((is_associative_container<std::map<int, double>>::value));
    EXPECT_TRUE((is_associative_container<std::unordered_map<int, double>>::value));
    EXPECT_TRUE((is_associative_container<std::unordered_multimap<int, double>>::value));

    EXPECT_FALSE((is_associative_container<std::set<int>>::value));
    EXPECT_FALSE((is_associative_container<std::set<std::pair<int, double>>>::value));
    EXPECT_FALSE((is_associative_container<std::vector<std::pair<int, double>>>::value));
}

// Sub-class map to check that find method is being properly used.

namespace {
    struct static_counter {
        static int count;
    } S;
    int static_counter::count = 0;

    template <typename K, typename V>
    struct check_map: std::map<K, V>, static_counter {
        using typename std::map<K, V>::value_type;
        using typename std::map<K, V>::iterator;
        using typename std::map<K, V>::const_iterator;

        check_map(): std::map<K, V>() {}
        check_map(std::initializer_list<value_type> init): std::map<K, V>(init) {}

        const_iterator find(const K& key) const {
            ++count;
            return std::map<K, V>::find(key);
        }

        iterator find(const K& key) {
            ++count;
            return std::map<K, V>::find(key);
        }
    };
}

TEST(maputil, value_by_key_map) {
    using util::value_by_key;
    using util::ptr_by_key;

    check_map<std::string, int> map_s2i = {
        {"fish", 4},
        {"sheep", 5},
    };

    S.count = 0;
    EXPECT_FALSE(value_by_key(map_s2i, "deer"));
    EXPECT_EQ(1, S.count);

    S.count = 0;
    auto r1 = value_by_key(map_s2i, "sheep");
    EXPECT_EQ(1, S.count);
    EXPECT_TRUE(r1);
    EXPECT_EQ(5, r1.value());

    auto p1 = ptr_by_key(map_s2i, "sheep");
    ASSERT_TRUE(p1);
    *p1 = 6;
    EXPECT_EQ(6, value_by_key(map_s2i, "sheep").value());

    S.count = 0;
    auto r2 = value_by_key(check_map<std::string, int>(map_s2i), "fish");
    EXPECT_EQ(1, S.count);
    EXPECT_TRUE(r2);
    EXPECT_EQ(4, r2.value());

    // Providing an explicit comparator should fall-back to serial search.

    S.count = 0;
    auto str_cmp = [](const std::string& k1, const char* k2) { return k1==k2; };
    auto r3 = value_by_key(map_s2i, "fish", str_cmp);
    EXPECT_EQ(0, S.count);
    EXPECT_TRUE(r3);
    EXPECT_EQ(4, r3.value());
}

TEST(maputil, value_by_key_sequence) {
    using util::value_by_key;
    using util::ptr_by_key;

    // Note: value_by_key returns the value of `get<1>` on the
    // entries in the map or sequence.

    using entry = std::tuple<int, std::string, char>;
    std::vector<entry> table = {
        entry(1, "one",   '1'),
        entry(3, "three", '3'),
        entry(5, "five",  '5')
    };

    EXPECT_FALSE(value_by_key(table, 2));

    auto r1 = value_by_key(table, 3);
    EXPECT_TRUE(r1);
    EXPECT_EQ("three"s, r1.value());

    auto p1 = ptr_by_key(table, 3);
    ASSERT_TRUE(p1);
    *p1 = "four";
    EXPECT_EQ("four"s, value_by_key(table, 3).value());

    auto r2 = value_by_key(std::move(table), 1);
    EXPECT_TRUE(r2);
    EXPECT_EQ("one", r2.value());
}

TEST(maputil, binary_search_index) {
    using util::binary_search_index;
    const int ns[] = {2, 3, 3, 5, 6, 113, 114, 114, 116};

    for (int x: {7, 1, 117}) {
        EXPECT_FALSE(binary_search_index(ns, x));
    }

    for (int x: {114, 3, 5, 2, 116}) {
        auto opti = binary_search_index(ns, x);
        ASSERT_TRUE(opti);
        EXPECT_EQ(x, ns[*opti]);
    }

    // With projection:
    const char* ss[] = {"bit", "short", "small", "longer", "very long", "also long", "sesquipedalian"};
    auto proj = [](const char* s) -> int { return std::strlen(s); };

    for (int x: {1, 4, 7, 10, 20}) {
        EXPECT_FALSE(binary_search_index(ss, x, proj));
    }

    for (int x: {3, 5, 6, 9, 14}) {
        auto opti = binary_search_index(ss, x, proj);
        ASSERT_TRUE(opti);
        EXPECT_EQ(x, (int)std::strlen(ss[*opti]));
    }
}
