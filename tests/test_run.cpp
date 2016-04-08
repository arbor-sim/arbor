#include "gtest.h"

#include "../src/cell.hpp"
#include "../src/fvm.hpp"

TEST(run, init)
{
    using namespace nest::mc;

    nest::mc::cell cell;

    cell.add_soma(12.6157/2.0);
    //auto& props = cell.soma()->properties;

    cell.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 200);

    EXPECT_EQ(cell.tree().num_segments(), 2u);

    /*
    for(auto &s : cell.segments()) {
        std::cout << "volume : " << s->volume()
                  << " area : " << s->area()
                  << " ratio : " << s->volume()/s->area() << std::endl;
    }
    */

    // in this context (i.e. attached to a segment on a high-level cell)
    // a mechanism is essentially a set of parameters
    // - the only "state" is that used to define parameters
    cell.soma()->add_mechanism(hh_parameters());

    auto& soma_hh = cell.soma()->mechanism("hh");

    soma_hh.set("gnabar", 0.12);
    soma_hh.set("gkbar", 0.036);
    soma_hh.set("gl", 0.0003);
    soma_hh.set("el", -54.3);

    // check that parameter values were set correctly
    EXPECT_EQ(cell.soma()->mechanism("hh").get("gnabar").value, 0.12);
    EXPECT_EQ(cell.soma()->mechanism("hh").get("gkbar").value, 0.036);
    EXPECT_EQ(cell.soma()->mechanism("hh").get("gl").value, 0.0003);
    EXPECT_EQ(cell.soma()->mechanism("hh").get("el").value, -54.3);


    cell.segment(1)->set_compartments(2);

    //using fvm_cell = fvm::fvm_cell<double, int>;
    //fvm_cell fvcell(cell);
    // print out the parameters if you want...
    //std::cout << soma_hh << "\n";
}

// test out the parameter infrastructure
TEST(run, parameters)
{
    nest::mc::parameter_list list("test");
    EXPECT_EQ(list.name(), "test");
    EXPECT_EQ(list.num_parameters(), 0);

    nest::mc::parameter p("a", 0.12, {0, 10});

    // add_parameter() returns a bool that indicates whether
    // it was able to successfull add the parameter
    EXPECT_EQ(list.add_parameter(std::move(p)), true);
    EXPECT_EQ(list.num_parameters(), 1u);

    // test in place construction of a parameter
    EXPECT_EQ(list.add_parameter({"b", -3.0}), true);
    EXPECT_EQ(list.num_parameters(), 2u);

    // check that adding a parameter that already exists returns false
    // and does not increase the number of parameters
    EXPECT_EQ(list.add_parameter({"b", -3.0}), false);
    EXPECT_EQ(list.num_parameters(), 2u);

    auto &parms = list.parameters();
    EXPECT_EQ(parms[0].name, "a");
    EXPECT_EQ(parms[0].value, 0.12);
    EXPECT_EQ(parms[0].range.min, 0);
    EXPECT_EQ(parms[0].range.max, 10);

    EXPECT_EQ(parms[1].name, "b");
    EXPECT_EQ(parms[1].value, -3);
    EXPECT_EQ(parms[1].range.has_lower_bound(), false);
    EXPECT_EQ(parms[1].range.has_upper_bound(), false);
}

