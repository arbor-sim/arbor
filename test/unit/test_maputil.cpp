#include "../gtest.h"

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>

#include <util/maputil.hpp>
#include <util/rangeutil.hpp>

#include "common.hpp"

using namespace arb;

using namespace std::string_literals;
using testing::nocopy;
using testing::nomove;

// TODO: Add unit tests for other new functionality in maputil.

TEST(maputil, keys) {
    {
        std::map<int, double> map = {{10, 2.0}, {3, 8.0}};
        std::vector<int> expected = {3, 10};
        std::vector<int> keys = util::assign_from(util::keys(map));
        EXPECT_EQ(expected, keys);
    }

    {
        struct cmp {
            bool operator()(const nocopy<int>& a, const nocopy<int>& b) const {
                return a.value<b.value;
            }
        };
        std::map<nocopy<int>, double, cmp> map;
        map.insert(std::pair<nocopy<int>, double>(11, 2.0));
        map.insert(std::pair<nocopy<int>, double>(2,  0.3));
        map.insert(std::pair<nocopy<int>, double>(2,  0.8));
        map.insert(std::pair<nocopy<int>, double>(5,  0.1));

        std::vector<int> expected = {2, 5, 11};
        std::vector<int> keys;
        for (auto& k: util::keys(map)) {
            keys.push_back(k.value);
        }
        EXPECT_EQ(expected, keys);
    }

    {
        std::unordered_multimap<int, double> map = {{3, 0.1}, {5, 0.4}, {11, 0.8}, {5, 0.2}};
        std::vector<int> expected = {3, 5, 5, 11};
        std::vector<int> keys = util::assign_from(util::keys(map));
        util::sort(keys);
        EXPECT_EQ(expected, keys);
    }
}

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

    template <typename X>
    constexpr bool is_optional_reference(X) { return false; }

    template <typename X>
    constexpr bool is_optional_reference(util::optional<X&>) { return true; }
}

TEST(maputil, value_by_key_map) {
    using util::value_by_key;

    check_map<std::string, int> map_s2i = {
        {"fish", 4},
        {"sheep", 5},
    };

    S.count = 0;
    EXPECT_FALSE(value_by_key(map_s2i, "deer"));
    EXPECT_EQ(1, S.count);

    // Should get an optional reference if argument is an lvalue.

    S.count = 0;
    auto r1 = value_by_key(map_s2i, "sheep");
    EXPECT_EQ(1, S.count);
    EXPECT_TRUE(r1);
    EXPECT_EQ(5, r1.value());
    EXPECT_TRUE(is_optional_reference(r1));
    r1.value() = 6;
    EXPECT_EQ(6, value_by_key(map_s2i, "sheep").value());

    // Should not get an optional reference if argument is an rvalue.

    S.count = 0;
    auto r2 = value_by_key(check_map<std::string, int>(map_s2i), "fish");
    EXPECT_EQ(1, S.count);
    EXPECT_TRUE(r2);
    EXPECT_EQ(4, r2.value());
    EXPECT_FALSE(is_optional_reference(r2));

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
    EXPECT_TRUE(is_optional_reference(r1));
    r1.value() = "four";
    EXPECT_EQ("four"s, value_by_key(table, 3).value());

    auto r2 = value_by_key(std::move(table), 1);
    EXPECT_TRUE(r2);
    EXPECT_EQ("one", r2.value());
    EXPECT_FALSE(is_optional_reference(r2));
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
