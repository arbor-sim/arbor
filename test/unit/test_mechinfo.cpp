#include <map>
#include <string>
#include <vector>

#include <arbor/cable_cell.hpp>

#include "../gtest.h"

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

    EXPECT_TRUE(p.count("a"));
    EXPECT_TRUE(p.count("b"));
    EXPECT_TRUE(p.count("d"));

    EXPECT_EQ(p["a"], m["a"]);
    EXPECT_EQ(p["b"], m["b"]);
    EXPECT_EQ(p["d"], m["d"]);
}
