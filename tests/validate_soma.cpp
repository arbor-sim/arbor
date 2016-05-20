#include <fstream>

#include "gtest.h"
#include "util.hpp"

#include "../src/cell.hpp"
#include "../src/fvm.hpp"

// based on hh/Neuron/steps_A.py
TEST(soma, resolutions)
{
    using namespace nest::mc;

    nest::mc::cell cell;

    // setup global state for the mechanisms
    nest::mc::mechanisms::setup_mechanism_helpers();

    // Soma with diameter 18.8um and HH channel
    auto soma = cell.add_soma(18.8/2.0);
    soma->mechanism("membrane").set("r_L", 123); // no effect for single compartment cell
    soma->add_mechanism(hh_parameters());

    // add stimulus to the soma
    //cell.add_stimulus({0,0}, {5., 3., 0.1});
    cell.add_stimulus({0,0.5}, {10., 100., 0.1});

    // make the lowered finite volume cell
    fvm::fvm_cell<double, int> model(cell);

    auto i =0;
    for(auto dt : {0.02, 0.01, 0.005, 0.002, 0.001}) {
        std::vector<std::vector<double>> results(2);

        // set initial conditions
        using memory::all;
        model.voltage()(all) = -65.;
        model.initialize(); // have to do this _after_ initial conditions are set

        // run the simulation
        auto tfinal =   120.; // ms
        int nt = tfinal/dt;
        results[0].push_back(0.);
        results[1].push_back(model.voltage()[0]);
        for(auto i=0; i<nt; ++i) {
            model.advance(dt);
            // save voltage at soma
            results[0].push_back((i+1)*dt);
            results[1].push_back(model.voltage()[0]);
        }
        write_vis_file("v_" + std::to_string(i) + ".dat", results);
        ++i;
    }
}

