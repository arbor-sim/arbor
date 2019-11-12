/*
 * A miniapp that demonstrates using an external spike source.
 * Actual miniapp that runs real arbor -- connect to real nest or nest proxy
 */

#include <arbor/version.hpp>

#ifndef ARB_MPI_ENABLED

#include <iostream>

int main() {
    std::cerr << "**** Only runs with ARB_MPI_ENABLED ***" << std::endl;
    return 1;
}

#else //ARB_MPI_ENABLED

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>

#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>
#include <sup/json_params.hpp>

#include "branch_cell.hpp"

#include <mpi.h>
#include <arborenv/with_mpi.hpp>

#include "parameters.hpp"
#include "mpiutil.hpp"
#include "branch_cell.hpp"

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

class ring_recipe: public arb::recipe {
public:
    ring_recipe(unsigned num_cells, cell_parameters params, double min_delay, int num_nest_cells):
        num_cells_(num_cells),
        cell_params_(params),
        min_delay_(min_delay),
        num_nest_cells_(num_nest_cells)
    {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        return branch_cell(gid, cell_params_);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override {
        return 1;
    }

    // The cell has one target synapse, which will be connected to cell gid-1.
    cell_size_type num_targets(cell_gid_type gid) const override {
        return 1;
    }

    // Each cell has one incoming connection from an external source.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        cell_gid_type src = num_cells_ + (gid%num_nest_cells_); // round robin
        return {arb::cell_connection({src, 0}, {gid, 0}, even_weight_, min_delay_)};
    }

    // No event generators.
    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        return {};
    }

    // There is one probe (for measuring voltage at the soma) on the cell.
    cell_size_type num_probes(cell_gid_type gid)  const override {
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        // Get the appropriate kind for measuring voltage.
        cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
        // Measure at the soma.
        arb::mlocation loc(0, 0.0);

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }

    arb::util::any get_global_properties(arb::cell_kind) const override {
        return gprop_;
    }

private:
    cell_size_type num_cells_;
    cell_parameters cell_params_;
    double min_delay_;
    float event_weight_ = 0.01;
    int num_nest_cells_;
    arb::cable_cell_global_properties gprop_;
};

struct cell_stats {
    using size_type = unsigned;
    cell_size_type ncells = 0;
    size_type nsegs = 0;
    size_type ncomp = 0;

    cell_stats(arb::recipe& r, comm_info& info) {
        int nranks, rank;
        MPI_Comm_rank(info.comm, &rank);
        MPI_Comm_size(info.comm, &nranks);
        ncells = r.num_cells();
        size_type cells_per_rank = ncells/nranks;
        size_type b = rank*cells_per_rank;
        size_type e = (rank==nranks-1)? ncells: (rank+1)*cells_per_rank;
        size_type nsegs_tmp = 0;
        size_type ncomp_tmp = 0;
        for (size_type i=b; i<e; ++i) {
            auto c = arb::util::any_cast<arb::cable_cell>(r.get_cell_description(i));
            nsegs_tmp += c.num_segments();
            ncomp_tmp += c.num_compartments();
        }
        MPI_Allreduce(&nsegs_tmp, &nsegs, 1, MPI_UNSIGNED, MPI_SUM, info.comm);
        MPI_Allreduce(&ncomp_tmp, &ncomp, 1, MPI_UNSIGNED, MPI_SUM, info.comm);
    }

    friend std::ostream& operator<<(std::ostream& o, const cell_stats& s) {
        return o << "cell stats: "
                 << s.ncells << " cells; "
                 << s.nsegs << " segments; "
                 << s.ncomp << " compartments.";
    }
};

// callback for external spikes
struct extern_callback {
    comm_info info;

    extern_callback(comm_info info): info(info) {}

    std::vector<arb::spike> operator()(arb::time_type t) {
        std::vector<arb::spike> local_spikes; // arbor processes send no spikes
        print_vec_comm("ARB-send", local_spikes, info.comm);
        static int step = 0;
        std::cerr << "ARB: step " << step++ << std::endl;
        auto global_spikes = gather_spikes(local_spikes, MPI_COMM_WORLD);
        print_vec_comm("ARB-recv", global_spikes, info.comm);

        return global_spikes;
    }
};

//
//  N ranks = Nn + Na
//      Nn = number of nest ranks
//      Na = number of arbor ranks
//
//  Nest  on COMM_WORLD [0, Nn)
//  Arbor on COMM_WORLD [Nn, N)
//

int main(int argc, char** argv) {
    try {
        arborenv::with_mpi guard(argc, argv, false);

        auto info = get_comm_info(true);

        arb::proc_allocation resources;
        if (auto nt = arbenv::get_env_num_threads()) {
            resources.num_threads = nt;
        }
        else {
            resources.num_threads = arbenv::thread_concurrency();
        }
        resources.gpu_id = arbenv::find_private_gpu(info.comm);
        auto context = arb::make_context(resources, info.comm);
        
        const bool root = arb::rank(context) == 0;
        std::cout << sup::mask_stream(root);

        // Print a banner with information about hardware configuration
        std::cout << "gpu:      " << (has_gpu(context)? "yes": "no") << "\n";
        std::cout << "threads:  " << num_threads(context) << "\n";
        std::cout << "mpi:      " << (has_mpi(context)? "yes": "no") << "\n";
        std::cout << "ranks:    " << num_ranks(context) << "\n" << std::endl;

        auto params = read_options(argc, argv);

#ifdef ARB_PROFILE_ENABLED
        arb::profile::profiler_initialize(context);
#endif

        arb::profile::meter_manager meters;
        meters.start(context);

        std::cout << "ARB: starting handshake" << std::endl;

        // hand shake #1: communicate cell populations
        broadcast((int)params.num_cells, MPI_COMM_WORLD, info.arbor_root);
        int num_nest_cells = broadcast(0,  MPI_COMM_WORLD, info.nest_root);

        std::cout << "ARB: num_nest_cells: " << num_nest_cells << std::endl;

        // Create an instance of our recipe.
        ring_recipe recipe(params.num_cells, params.cell, params.min_delay, num_nest_cells);
        cell_stats stats(recipe, info);
        std::cout << stats << std::endl;

        auto decomp = arb::partition_load_balance(recipe, context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);

        // hand shake #2: min delay
        float arb_comm_time = sim.min_delay()/2;
        broadcast(arb_comm_time, MPI_COMM_WORLD, info.arbor_root);
        float nest_comm_time = broadcast(0.f, MPI_COMM_WORLD, info.nest_root);
        auto min_delay = sim.min_delay(nest_comm_time*2);
        std::cout << "ARB: min_delay=" << min_delay << std::endl;

        float delta = min_delay/2;
        float sim_duration = params.duration;
        unsigned steps = sim_duration/delta;
        if (steps*delta < sim_duration) ++steps;

        //hand shake #3: steps
        broadcast(steps, MPI_COMM_WORLD, info.arbor_root);

        // Set up recording of spikes to a vector on the root process.
        std::vector<arb::spike> recorded_spikes;
        if (root) {
            sim.set_global_spike_callback(
                [&recorded_spikes](const std::vector<arb::spike>& spikes) {
                    print_vec_comm("ARB", spikes);
                    recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                });
        }

        // Define the external spike source callback
        sim.set_external_spike_callback(extern_callback(info));

        meters.checkpoint("model-init", context);

        std::cout << "ARB: running simulation" << std::endl;
        // Run the simulation for 100 ms, with time steps of 0.025 ms.
        sim.run(params.duration, 0.025);

        meters.checkpoint("model-run", context);

        auto ns = sim.num_spikes();

        // Write spikes to file
        if (root) {
            std::cout << "\nARB: " << ns << " spikes generated at rate of "
                      << params.duration/ns << " ms between spikes\n";
            std::ofstream fid("spikes.gdf");
            if (!fid.good()) {
                std::cerr << "ARB: Warning: unable to open file spikes.gdf for spike output\n";
            }
            else {
                char linebuf[45];
                for (auto spike: recorded_spikes) {
                    auto n = std::snprintf(
                        linebuf, sizeof(linebuf), "%u %.4f\n",
                        unsigned{spike.source.gid}, float(spike.time));
                    fid.write(linebuf, n);
                }
            }
        }

        auto report = arb::profile::make_meter_report(meters, context);
        std::cout << report;
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp:\n" << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

#endif //ARB_MPI_ENABLED
