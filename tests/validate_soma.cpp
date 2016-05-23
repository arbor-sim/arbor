#include <fstream>

#include "gtest.h"
#include "util.hpp"

#include <cell.hpp>
#include <fvm.hpp>

#include <json/src/json.hpp>

// based on hh/Neuron/steps_A.py
TEST(soma, resolutions)
{
    using namespace nest::mc;
    using namespace nlohmann;

    nest::mc::cell cell;

    // setup global state for the mechanisms
    nest::mc::mechanisms::setup_mechanism_helpers();

    // Soma with diameter 18.8um and HH channel
    auto soma = cell.add_soma(18.8/2.0);
    soma->mechanism("membrane").set("r_L", 123); // no effect for single compartment cell
    soma->add_mechanism(hh_parameters());

    // add stimulus to the soma
    cell.add_stimulus({0,0.5}, {10., 100., 0.1});

    // make the lowered finite volume cell
    fvm::fvm_cell<double, int> model(cell);

    // load data from file
    std::string input_name = "../nrn/soma.json";
    json  cell_data;
    {
        auto fid = std::ifstream(input_name);
        if(!fid.is_open()) {
            std::cerr << "error : unable to open file " << input_name
                      << " : run the validation generation script first\n";
            return;
        }

        try {
            fid >> cell_data;
        }
        catch (...) {
            std::cerr << "error : incorrectly formatted json file " << input_name << "\n";
            return;
        }
    }

    for(auto& run : cell_data) {
        std::vector<double> v;
        double dt = run["dt"];

        // set initial conditions
        using memory::all;
        model.voltage()(all) = -65.;
        model.initialize(); // have to do this _after_ initial conditions are set

        // run the simulation
        auto tfinal =   120.; // ms
        int nt = tfinal/dt;
        v.push_back(model.voltage()[0]);
        for(auto i=0; i<nt; ++i) {
            model.advance(dt);
            // save voltage at soma
            v.push_back(model.voltage()[0]);
        }

        // get the spike times from the NEST MC and NEURON simulations respectively
        auto nst_spike_times = find_spikes(v, 0., dt);
        auto nrn_spike_times = run["spikes"].get<std::vector<double>>();
        auto comparison = compare_spikes(nst_spike_times, nrn_spike_times);

        // Assert that relative error is less than 1%.
        // For a 100 ms simulation this asserts that the difference between NEST and NEURON
        // at a given time step size is less than 1 ms.
        EXPECT_TRUE(comparison.max_relative_error()*100. < 1.);
    }
}

