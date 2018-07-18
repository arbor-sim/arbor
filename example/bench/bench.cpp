/*
 * Miniapp that uses the artificial benchmark cell type to test
 * the simulator infrastructure.
 */
#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/profile/meter_manager.hpp>
#include <arbor/common_types.hpp>
#include <arbor/distributed_context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>

#include <aux/ioutil.hpp>
#include <aux/json_meter.hpp>
#ifdef ARB_MPI_ENABLED
#include <aux/with_mpi.hpp>
#endif

#include "parameters.hpp"
#include "recipe.hpp"

namespace profile = arb::profile;

int main(int argc, char** argv) {
    try {
        arb::distributed_context context;
#ifdef ARB_HAVE_MPI
        aux::with_mpi guard(&argc, &argv);
        context = mpi_context(MPI_COMM_WORLD);
#endif
        const bool is_root =  context.id()==0;

        std::cout << aux::mask_stream(is_root);

        bench_params params = read_options(argc, argv);

        std::cout << params << "\n";

        profile::meter_manager meters(&context);
        meters.start();

        // Create an instance of our recipe.
        bench_recipe recipe(params);
        meters.checkpoint("recipe-build");

        // Make the domain decomposition for the model
        auto local = arb::local_allocation();
        auto decomp = arb::partition_load_balance(recipe, local, &context);
        meters.checkpoint("domain-decomp");

        // Construct the model.
        arb::simulation sim(recipe, decomp, &context);
        meters.checkpoint("model-build");

        // Run the simulation for 100 ms, with time steps of 0.01 ms.
        sim.run(params.duration, 0.01);
        meters.checkpoint("model-run");

        // write meters
        auto report = profile::make_meter_report(meters);
        std::cout << report << "\n";

        if (is_root==0) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << aux::to_json(report) << "\n";
        }

        // output profile and diagnostic feedback
        auto summary = profile::profiler_summary();
        std::cout << summary << "\n";

        std::cout << "there were " << sim.num_spikes() << " spikes\n";
    }
    catch (std::exception& e) {
        std::cerr << "exception caught running benchmark miniapp:\n" << e.what() << std::endl;
    }
}
