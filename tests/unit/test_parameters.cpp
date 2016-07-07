#include <fstream>

#include "gtest.h"
#include "util.hpp"

#include "../src/parameter_list.hpp"

// test out the parameter infrastructure
TEST(parameters, setting)
{
    nest::mc::parameter_list list("test");
    EXPECT_EQ(list.name(), "test");
    EXPECT_EQ(list.num_parameters(), 0);

    nest::mc::parameter p("a", 0.12, {0, 10});

    // add_parameter() returns a bool that indicates whether
    // it was able to successfull add the parameter
    EXPECT_TRUE(list.add_parameter(std::move(p)));
    EXPECT_EQ(list.num_parameters(), 1);

    // test in place construction of a parameter
    EXPECT_TRUE(list.add_parameter({"b", -3.0}));
    EXPECT_EQ(list.num_parameters(), 2);

    // check that adding a parameter that already exists returns false
    // and does not increase the number of parameters
    EXPECT_FALSE(list.add_parameter({"b", -3.0}));
    EXPECT_EQ(list.num_parameters(), 2);

    auto &parms = list.parameters();
    EXPECT_EQ(parms[0].name, "a");
    EXPECT_EQ(parms[0].value, 0.12);
    EXPECT_EQ(parms[0].range.min, 0);
    EXPECT_EQ(parms[0].range.max, 10);

    EXPECT_EQ(parms[1].name, "b");
    EXPECT_EQ(parms[1].value, -3);
    EXPECT_FALSE(parms[1].range.has_lower_bound());
    EXPECT_FALSE(parms[1].range.has_upper_bound());
}
