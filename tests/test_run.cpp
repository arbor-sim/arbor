#include "gtest.h"

#include "../src/cell.hpp"
#include "../src/fvm.hpp"

// based on hh/Neuron/steps_A.py
TEST(run, single_compartment)
{
    using namespace nest::mc;

    nest::mc::cell cell;

    // setup global state for the mechanisms
    nest::mc::mechanisms::setup_mechanism_helpers();

    // Soma with diameter 18.8um -> 1.88e-3 cm and HH channel
    auto soma = cell.add_soma(1.88e-3/2.0);
    soma->mechanism("membrane").set("r_L", 123); // no effect for single compartment cell
    soma->add_mechanism(hh_parameters());

    std::cout << soma->mechanism("membrane");

    // make the lowered finite volume cell
    fvm::fvm_cell<double, int> model(cell);

    //std::cout << "CV areas " << model.cv_areas() << "\n";

    //std::cout << "-----------------------------\n";

    // set initial conditions
    using memory::all;
    model.voltage()(all) = -65.;
    model.initialize(); // have to do this _after_ initial conditions are set

    //std::cout << "-----------------------------\n";

    // run the simulation
    auto dt     =   0.02; // ms
    //auto tfinal =   0.10; // ms
    auto tfinal =   10.; // ms
    int nt = tfinal/dt;
    std::vector<double> result;
    result.push_back(model.voltage()[0]);
    for(auto i=0; i<nt; ++i) {
        model.advance(dt);
        result.push_back(model.voltage()[0]);
    }
    std::cout << "took " << nt << " time steps" << std::endl;

    {
        std::ofstream fid("v.dat");
        auto t = 0.;
        for(auto v:result) {
            fid << t << " " << v << "\n";
            t += dt;
        }
    }
}

TEST(run, ball_and_stick)
{
    using namespace nest::mc;

    nest::mc::cell cell;

    // setup global state for the mechanisms
    nest::mc::mechanisms::setup_mechanism_helpers();

    // Soma with diameter 12.6157 and HH channel
    auto soma = cell.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    // add dendrite with passive channel and 10 compartments
    auto dendrite = cell.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 200);
    dendrite->set_compartments(5);
    dendrite->add_mechanism(pas_parameters());

    // make the lowered finite volume cell
    fvm::fvm_cell<double, int> model(cell);

    std::cout << "CV areas " << model.cv_areas() << "\n";

    // set initial conditions
    using memory::all;
    model.voltage()(all) = -65.;

    // run the simulation
    auto dt = 0.02; // ms
    model.advance(dt);

    std::cout << model.voltage() << "\n";
}

TEST(run, cable)
{
    using namespace nest::mc;

    nest::mc::cell cell;

    // setup global state for the mechanisms
    nest::mc::mechanisms::setup_mechanism_helpers();

    cell.add_soma(6e-4); // 6um in cm

    // 1um radius and 4mm long, all in cm
    cell.add_cable(0, segmentKind::dendrite, 1e-4, 1e-4, 4e-1);
    cell.add_cable(0, segmentKind::dendrite, 1e-4, 1e-4, 4e-1);

    //std::cout << cell.segment(1)->area() << " is the area\n";
    EXPECT_EQ(cell.model().tree.num_segments(), 3u);

    // add passive to all 3 segments in the cell
    for(auto& seg :cell.segments()) {
        seg->add_mechanism(pas_parameters());
    }

    cell.soma()->add_mechanism(hh_parameters());
    cell.segment(2)->add_mechanism(hh_parameters());

    auto& soma_hh = cell.soma()->mechanism("hh");

    soma_hh.set("gnabar", 0.12);
    soma_hh.set("gkbar", 0.036);
    soma_hh.set("gl", 0.0003);
    soma_hh.set("el", -54.387);

    cell.segment(1)->set_compartments(4);
    cell.segment(2)->set_compartments(4);

    using fvm_cell = fvm::fvm_cell<double, int>;
    fvm_cell fvcell(cell);
    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 9u);

    fvcell.setup_matrix(0.02);
    EXPECT_EQ(fvcell.cv_areas().size(), J.size());

    // inspect ion channels
    /*
    std::cout << "ion na index : " << fvcell.ion_na().node_index() << "\n";
    std::cout << "ion ca index : " << fvcell.ion_ca().node_index() << "\n";
    std::cout << "ion k  index : " << fvcell.ion_k().node_index() << "\n";

    std::cout << "ion na E : " << fvcell.ion_na().reversal_potential() << "\n";
    std::cout << "ion ca E : " << fvcell.ion_ca().reversal_potential() << "\n";
    std::cout << "ion k  E : " << fvcell.ion_k().reversal_potential() << "\n";
    */

    //auto& cable_parms = cell.segment(1)->mechanism("membrane");
    //std::cout << soma_hh << std::endl;
    //std::cout << cable_parms << std::endl;

    //std::cout << "l " << J.l() << "\n";
    //std::cout << "d " << J.d() << "\n";
    //std::cout << "u " << J.u() << "\n";
    //std::cout << "p " << J.p() << "\n";

    J.rhs()(memory::all) = 1.;
    J.rhs()[0] = 10.;

    J.solve();
}

TEST(run, init)
{
    using namespace nest::mc;

    nest::mc::cell cell;

    // setup global state for the mechanisms
    nest::mc::mechanisms::setup_mechanism_helpers();

    cell.add_soma(12.6157/2.0);
    //auto& props = cell.soma()->properties;

    cell.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 200);

    const auto m = cell.model();
    EXPECT_EQ(m.tree.num_segments(), 2u);

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

    cell.segment(1)->set_compartments(10);

    using fvm_cell = fvm::fvm_cell<double, int>;
    fvm_cell fvcell(cell);
    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 11u);

    fvcell.setup_matrix(0.01);
    //std::cout << "areas " << fvcell.cv_areas() << "\n";

    J.rhs()(memory::all) = 1.;
    J.rhs()[0] = 10.;

    J.solve();

    //std::cout << "x" << J.rhs() << "\n";
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
    EXPECT_EQ(list.num_parameters(), 1);

    // test in place construction of a parameter
    EXPECT_EQ(list.add_parameter({"b", -3.0}), true);
    EXPECT_EQ(list.num_parameters(), 2);

    // check that adding a parameter that already exists returns false
    // and does not increase the number of parameters
    EXPECT_EQ(list.add_parameter({"b", -3.0}), false);
    EXPECT_EQ(list.num_parameters(), 2);

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

