#include "../gtest.h"

#include <map>
#include <unordered_map>
#include <vector>
#include <utility>

#include <util/maputil.hpp>
#include <util/rangeutil.hpp>

#include "common.hpp"

using namespace arb;

using testing::nocopy;
using testing::nomove;

// TODO: Add unit tests for other new functionality in maputil.

TEST(range, keys) {
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

