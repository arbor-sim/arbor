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

        std::cout << v << "\n";
        auto spike_times = find_spikes(v, 0., dt);
        std::cout << "dt " << dt << " : spikes " << spike_times << "\n";
    }
}

