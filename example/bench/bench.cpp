/*
 * Miniapp that uses the artificial benchmark cell type to test
 * the simulator infrastructure.
 */
#include <fstream>
#include <iomanip>
#include <iostream>

#include <json/json.hpp>

#include <communication/global_policy.hpp>
#include <common_types.hpp>
#include <hardware/node_info.hpp>
#include <load_balance.hpp>
#include <profiling/meter_manager.hpp>
#include <simulation.hpp>
#include <util/ioutil.hpp>

#include "recipe.hpp"

using namespace arb;

int main() {
    std::cout << util::mask_stream(communication::global_policy::id()==0);

    bench_params params;
    params.name = "test";
    params.num_cells = 100;
    params.tfinal = 100;
    params.cell.spike_freq_hz = 20;
    params.cell.us_per_ms = 20;
    params.network.fan_in = 10000;
    params.network.min_delay = 10;

    std::cout << params << "\n";

    try {
        bench_params p("input.json");
    }
    catch (std::exception& e) {
        std::cout << e.what() << "\n";
    }

    try {
        util::meter_manager meters;
        meters.start();

        // Create an instance of our recipe.
        bench_recipe recipe(params);
        meters.checkpoint("recipe-build");

        // Make the domain decomposition for the model
        auto node = arb::hw::get_node_info();
        auto decomp = arb::partition_load_balance(recipe, node);
        meters.checkpoint("domain-decomp");

        // Construct the model.
        arb::simulation sim(recipe, decomp);
        meters.checkpoint("model-build");

        // Run the simulation for 100 ms, with time steps of 0.01 ms.
        sim.run(params.tfinal, 0.01);
        meters.checkpoint("model-run");

        // write meters
        auto report = util::make_meter_report(meters);
        std::cout << report << "\n";

        if (communication::global_policy::id()==0) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << util::to_json(report) << "\n";
        }

        // output profile and diagnostic feedback
        auto profile = util::profiler_summary();
        std::cout << "\n" << profile << "\n\n";

        unsigned expected_spikes = params.tfinal/1000. * params.num_cells * params.cell.spike_freq_hz;
        std::cout << "\nthere were " << sim.num_spikes() << " spikes (expected " << expected_spikes << ")\n";
    }
    catch (std::exception e) {
        std::cerr << "exception caught running benchmark miniapp:\n" << e.what() << std::endl;
    }
}
