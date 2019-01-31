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


#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
#endif

#include "parameters.hpp"
#include "recipe.hpp"

namespace profile = arb::profile;

int main(int argc, char** argv) {
    bool is_root = true;

    try {
        arb::proc_allocation resources;
        if (auto nt = arbenv::get_env_num_threads()) {
            resources.num_threads = nt;
        }
        else {
            resources.num_threads = arbenv::thread_concurrency();
        }

#ifdef ARB_MPI_ENABLED
        arbenv::with_mpi guard(argc, argv, false);
        resources.gpu_id = arbenv::find_private_gpu(MPI_COMM_WORLD);
        auto context = arb::make_context(resources, MPI_COMM_WORLD);
        is_root = arb::rank(context) == 0;
#else
        resources.gpu_id = arbenv::default_gpu();
        auto context = arb::make_context(resources);
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
