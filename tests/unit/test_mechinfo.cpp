#include <map>
#include <string>
#include <vector>

#include "mechinfo.hpp"

#include "../gtest.h"
#include "../test_util.hpp"

// TODO: expand tests when we have exported mechanism schemata
// from modcc.

using namespace arb;

TEST(mechanism_spec, setting) {
    mechanism_spec m("foo");

    m.set("a", 3.2);
    m.set("b", 4.3);

    auto dflt = m["c"];
    EXPECT_EQ(0., dflt); // note: 0 default is artefact of dummy schema

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
