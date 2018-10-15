/*
 * Miniapp that uses the artificial benchmark cell type to test
 * the simulator infrastructure.
 */
#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/profile/meter_manager.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/version.hpp>


#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>
#ifdef ARB_MPI_ENABLED
#include <sup/with_mpi.hpp>
#endif

#include "parameters.hpp"
#include "recipe.hpp"

namespace profile = arb::profile;

int main(int argc, char** argv) {
    bool is_root = true;

    try {
#ifdef ARB_MPI_ENABLED
        sup::with_mpi guard(argc, argv, false);
        auto context = arb::make_context(arb::proc_allocation(), MPI_COMM_WORLD);
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            is_root = rank==0;
        }
#else
        auto context = arb::make_context();
#endif
#ifdef ARB_PROFILE_ENABLED
        profile::profiler_initialize(context);
#endif

        std::cout << sup::mask_stream(is_root);

        bench_params params = read_options(argc, argv);

        std::cout << params << "\n";

        profile::meter_manager meters;
        meters.start(context);

        // Create an instance of our recipe.
        bench_recipe recipe(params);
        meters.checkpoint("recipe-build", context);

        // Make the domain decomposition for the model
        auto decomp = arb::partition_load_balance(recipe, context);
        meters.checkpoint("domain-decomp", context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);
        meters.checkpoint("model-build", context);

        // Run the simulation for 100 ms, with time steps of 0.01 ms.
        sim.run(params.duration, 0.01);
        meters.checkpoint("model-run", context);

        // write meters
        auto report = profile::make_meter_report(meters, context);
        std::cout << report << "\n";

        if (is_root) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << sup::to_json(report) << "\n";
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
