#include <cmath>
#include <exception>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/execution_context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/sampling.hpp>
#include <arbor/schedule.hpp>
#include <arbor/simulation.hpp>
#include <arbor/util/any.hpp>
#include <arbor/version.hpp>


#include <aux/ioutil.hpp>
#include <aux/json_meter.hpp>
#include <aux/path.hpp>
#include <aux/spike_emitter.hpp>
#include <aux/strsub.hpp>
#ifdef ARB_MPI_ENABLED
#include <aux/with_mpi.hpp>
#endif

#include "io.hpp"
#include "dryrun_recipes.hpp"

using namespace arb;

using util::any_cast;

void banner(proc_allocation, const execution_context&);
std::unique_ptr<recipe> make_recipe(const io::cl_options&, const probe_distribution&, const execution_context&);

int main(int argc, char** argv) {
    // default serial context
    execution_context context;

    try {
#ifdef ARB_MPI_ENABLED
        aux::with_mpi guard(argc, argv, false);
        context.distributed = mpi_context(MPI_COMM_WORLD);
#endif
#ifdef ARB_PROFILE_ENABLED
        profile::profiler_initialize(context.thread_pool);
#endif
        // read parameters
        io::cl_options options = io::read_options(argc, argv, context.distributed->id()==0);

        if (options.dry_run_ranks) {
            context.distributed = dry_run_context(options.dry_run_ranks);
        }

        profile::meter_manager meters(context.distributed);
        meters.start();

        std::cout << aux::mask_stream(context.distributed->id()==0);

        // TODO: add dry run mode

        // Use a node description that uses the number of threads used by the
        // threading back end, and 1 gpu if available.
        proc_allocation nd = local_allocation(context);
        nd.num_gpus = nd.num_gpus>=1? 1: 0;
        banner(nd, context);

        meters.checkpoint("setup");

        // determine what to attach probes to
        probe_distribution pdist;
        pdist.proportion = options.probe_ratio;
        pdist.all_segments = !options.probe_soma_only;

        auto recipe = make_recipe(options, pdist, context);
        context.distributed->set_num_cells(recipe->num_cells());

        auto decomp = partition_load_balance(*recipe, nd, context);
        simulation sim(*recipe, decomp, context);

        // Specify event binning/coalescing.
        auto binning_policy =
            options.bin_dt==0? binning_kind::none:
            options.bin_regular? binning_kind::regular:
            binning_kind::following;

        sim.set_binning_policy(binning_policy, options.bin_dt);

        // Initialize the spike exporting interface
        std::fstream spike_out;
        if (options.spike_file_output) {
            using std::ios_base;

            auto rank = context.distributed->id();
            aux::path p = options.output_path;
            p /= aux::strsub("%_%.%", options.file_name, rank, options.file_extension);

            if (options.single_file_per_rank) {
                spike_out = aux::open_or_throw(p, ios_base::out, !options.over_write);
                sim.set_local_spike_callback(aux::spike_emitter(spike_out));
            }
            else if (rank==0) {
                spike_out = aux::open_or_throw(p, ios_base::out, !options.over_write);
                sim.set_global_spike_callback(aux::spike_emitter(spike_out));
            }
        }

        meters.checkpoint("model-init");

        // run model
        sim.run(options.tfinal, options.dt);

        meters.checkpoint("model-simulate");

        // output profile and diagnostic feedback
        auto profile = profile::profiler_summary();
        std::cout << profile << "\n";
        std::cout << "\nthere were " << sim.num_spikes() << " spikes\n";

        auto report = profile::make_meter_report(meters);
        std::cout << report;
        if (context.distributed->id()==0) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << aux::to_json(report) << "\n";
        }
    }
    catch (io::usage_error& e) {
        // only print usage/startup errors on master
        std::cerr << aux::mask_stream(context.distributed->id()==0);
        std::cerr << e.what() << "\n";
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    return 0;
}

void banner(proc_allocation nd, const execution_context& ctx) {
    std::cout << "==========================================\n";
    std::cout << "  Arbor miniapp\n";
    std::cout << "  - distributed : " << ctx.distributed->size()
              << " (" << ctx.distributed->name() << ")\n";
    std::cout << "  - threads     : " << nd.num_threads << "\n";
    std::cout << "  - gpus        : " << nd.num_gpus << "\n";
    std::cout << "==========================================\n";
}

std::unique_ptr<recipe> make_recipe(const io::cl_options& options, const probe_distribution& pdist, const execution_context& ctx) {
    basic_recipe_param p;

    p.num_compartments = options.compartments_per_segment;
    p.num_synapses = options.synapses_per_cell;
    p.synapse_type = options.syn_type;

    return make_basic_rgraph_symmetric_recipe(options.cells, ctx.distributed->size(), p, pdist);
}