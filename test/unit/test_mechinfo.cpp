#include <string>
#include <vector>

#include <arbor/cable_cell.hpp>

#include <gtest/gtest.h>
#include "unit_test_catalogue.hpp"

// TODO: This test is really checking part of the recipe description
// for cable1d cells, so move it there. Make actual tests for mechinfo
// here!

using namespace arb;

TEST(mechanism_desc, setting) {
    mechanism_desc m("foo");

    m.set("a", 3.2);
    m.set("b", 4.3);

    EXPECT_EQ(3.2, m["a"]);
    EXPECT_EQ(4.3, m["b"]);

    m["b"] = 5.4;
    m["d"] = 6.5;

    EXPECT_EQ(3.2, m["a"]);
    EXPECT_EQ(5.4, m["b"]);
    EXPECT_EQ(6.5, m["d"]);

    // Check values() method is faithful:
    auto p = m.values();
    EXPECT_EQ(3u, p.size());

    auto count = [](const auto& nm, const auto& kvs) {
        int res = 0;
        for (const auto& [k,v]: kvs) {
            if (nm == k) res += 1;
        }
        return res;
    };

    EXPECT_TRUE(count("a", p));
    EXPECT_TRUE(count("b", p));
    EXPECT_TRUE(count("d", p));

    auto find = [](const auto& nm, const auto& kvs) -> double {
        for (const auto& [k,v]: kvs) {
            if (nm == k) return v;
        }
        return std::nan("");
    };

    EXPECT_EQ(find("a", p), m["a"]);
    EXPECT_EQ(find("b", p), m["b"]);
    EXPECT_EQ(find("d", p), m["d"]);
}

TEST(mechanism_desc, linearity) {
    {
        auto cat = arb::global_default_catalogue();
        EXPECT_TRUE(cat["expsyn"].linear);
        EXPECT_TRUE(cat["exp2syn"].linear);
    }
    {
        auto cat = make_unit_test_catalogue();
        EXPECT_FALSE(cat["non_linear"].linear);
    }
}
