/*
 * Miniapp that uses the artificial benchmark cell type to test
 * the simulator infrastructure.
 */
#include <fstream>
#include <iomanip>
#include <iostream>

#include <json/json.hpp>

#include <common_types.hpp>
#include <communication/distributed_context.hpp>
#include <hardware/node_info.hpp>
#include <load_balance.hpp>
#include <profiling/meter_manager.hpp>
#include <simulation.hpp>
#include <util/ioutil.hpp>

#include "recipe.hpp"

using namespace arb;

int main(int argc, char** argv) {
    try {
        distributed_context context;
        #ifdef ARB_HAVE_MPI
        mpi::scoped_guard guard(&argc, &argv);
        context = mpi_context(MPI_COMM_WORLD);
        #endif
        const bool is_root =  context.id()==0;

        std::cout << util::mask_stream(is_root);

        bench_params params = read_options(argc, argv);

        std::cout << params << "\n";

        util::meter_manager meters(&context);
        meters.start();

        // Create an instance of our recipe.
        bench_recipe recipe(params);
        meters.checkpoint("recipe-build");

        // Make the domain decomposition for the model
        auto node = arb::hw::get_node_info();
        auto decomp = arb::partition_load_balance(recipe, node, &context);
        meters.checkpoint("domain-decomp");

        // Construct the model.
        arb::simulation sim(recipe, decomp, &context);
        meters.checkpoint("model-build");

        // Run the simulation for 100 ms, with time steps of 0.01 ms.
        sim.run(params.duration, 0.01);
        meters.checkpoint("model-run");

        // write meters
        auto report = util::make_meter_report(meters);
        std::cout << report << "\n";

        if (is_root==0) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << util::to_json(report) << "\n";
        }

        // output profile and diagnostic feedback
        auto profile = util::profiler_summary();
        std::cout << profile << "\n";

        std::cout << "there were " << sim.num_spikes() << " spikes\n";
    }
    catch (std::exception& e) {
        std::cerr << "exception caught running benchmark miniapp:\n" << e.what() << std::endl;
    }
}
