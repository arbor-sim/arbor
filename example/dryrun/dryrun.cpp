/*
 * A miniapp that demonstrates how to use dry_run mode
 *
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/symmetric_recipe.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>
#include <sup/json_params.hpp>

#include "branch_cell.hpp"

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
#endif

struct run_params {
    std::string name = "default";
    bool dry_run = false;
    unsigned num_cells_per_rank = 10;
    unsigned num_ranks = 1;
    double min_delay = 10;
    double duration = 100;
    cell_parameters cell;
};

run_params read_options(int argc, char** argv);

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

// Generate a cell.
arb::cable_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params);

class tile_desc: public arb::tile {
public:
    tile_desc(unsigned num_cells, unsigned num_tiles, cell_parameters params, unsigned min_delay):
            num_cells_(num_cells),
            num_tiles_(num_tiles),
            cell_params_(params),
            min_delay_(min_delay)
    {}

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    cell_size_type num_tiles() const override {
        return num_tiles_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        return branch_cell(gid, cell_params_);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }

    arb::util::any get_global_properties(arb::cell_kind) const override {
        arb::cable_cell_global_properties gprop;
        gprop.default_parameters = arb::neuron_parameter_defaults;
        return gprop;
    }

    // Each cell has one spike detector (at the soma).
    cell_size_type num_sources(cell_gid_type gid) const override {
        return 1;
    }

    // The cell has one target synapse
    cell_size_type num_targets(cell_gid_type gid) const override {
        return 1;
    }

    // Each cell has one incoming connection, from any cell in the network spanning all ranks:
    // src gid in {0, ..., num_cells_*num_tiles_ - 1}.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        std::uniform_int_distribution<cell_gid_type> source_distribution(0, num_cells_*num_tiles_ - 2);

        auto src_gen = std::mt19937(gid);
        auto src = source_distribution(src_gen);
        if (src>=gid) ++src;

        return {arb::cell_connection({src, 0}, {gid, 0}, event_weight_, min_delay_)};
    }

    // Return an event generator on every 20th gid. This function needs to generate events
    // for ALL cells on ALL ranks. This is because the symmetric recipe can not easily
    // translate the src gid of an event generator
    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        std::vector<arb::event_generator> gens;
        if (gid%20 == 0) {
            gens.push_back(arb::explicit_generator(arb::pse_vector{{{gid, 0}, 0.1, 1.0}}));
        }
        return gens;
    }

    // There is one probe (for measuring voltage at the soma) on the cell.
    cell_size_type num_probes(cell_gid_type gid)  const override {
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        // Get the appropriate kind for measuring voltage.
        cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
        // Measure at the soma.
        arb::mlocation loc{0, 0.0};

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }

private:
    cell_size_type num_cells_;
    cell_size_type num_tiles_;
    cell_parameters cell_params_;
    double min_delay_;
    float event_weight_ = 0.01;
};

int main(int argc, char** argv) {
    try {
#ifdef ARB_MPI_ENABLED
        arbenv::with_mpi guard(argc, argv, false);
#endif
        bool root = true;
        auto params = read_options(argc, argv);

        auto resources = arb::proc_allocation();
        auto ctx = arb::make_context(resources);

        if (params.dry_run) {
            ctx = arb::make_context(resources, arb::dry_run_info(params.num_ranks, params.num_cells_per_rank));
        }
#ifdef ARB_MPI_ENABLED
        else {
            ctx = arb::make_context(resources, MPI_COMM_WORLD);
            {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                root = rank==0;
            }
        }
#endif
        arb_assert(arb::num_ranks(ctx)==params.num_ranks);


#ifdef ARB_PROFILE_ENABLED
        arb::profile::profiler_initialize(ctx);
#endif
        std::cout << sup::mask_stream(root);

        // Print a banner with information about hardware configuration
        std::cout << "gpu:      " << (has_gpu(ctx)? "yes": "no") << "\n";
        std::cout << "threads:  " << num_threads(ctx) << "\n";
        std::cout << "mpi:      " << (has_mpi(ctx)? "yes": "no") << "\n";
        std::cout << "ranks:    " << num_ranks(ctx) << "\n" << std::endl;

        std::cout << "run mode: " << distribution_type(ctx) << "\n";

        arb::profile::meter_manager meters;
        meters.start(ctx);

        // Create an instance of our tile and use it to make a symmetric_recipe.
        auto tile = std::make_unique<tile_desc>(params.num_cells_per_rank,
                params.num_ranks, params.cell, params.min_delay);
        arb::symmetric_recipe recipe(std::move(tile));

        auto decomp = arb::partition_load_balance(recipe, ctx);

        // Construct the model.
        arb::simulation sim(recipe, decomp, ctx);

        // Set up recording of spikes to a vector on the root process.
        std::vector<arb::spike> recorded_spikes;
        if (root) {
            sim.set_global_spike_callback(
                    [&recorded_spikes](const std::vector<arb::spike>& spikes) {
                        recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                    });
        }

        meters.checkpoint("model-init", ctx);

        // Run the simulation for 100 ms, with time steps of 0.025 ms.
        sim.run(params.duration, 0.025);

        meters.checkpoint("model-run", ctx);

        auto ns = sim.num_spikes();
        std::cout << "\n" << ns << " spikes generated at rate of "
                  << params.duration/ns << " ms between spikes\n\n";

        // Write spikes to file
        if (root) {
            std::ofstream fid("spikes.gdf");
            if (!fid.good()) {
                std::cerr << "Warning: unable to open file spikes.gdf for spike output\n";
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

        auto profile = arb::profile::profiler_summary();
        std::cout << profile << "\n";

        auto report = arb::profile::make_meter_report(meters, ctx);
        std::cout << report;
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}

run_params read_options(int argc, char** argv) {
    using sup::param_from_json;

    run_params params;
    if (argc<2) {
        std::cout << "Using default parameters.\n";
        return params;
    }
    if (argc>2) {
        throw std::runtime_error("More than command line one option not permitted.");
    }

    std::string fname = argv[1];
    std::cout << "Loading parameters from file: " << fname << "\n";
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("Unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    json << f;

    param_from_json(params.name, "name", json);
    param_from_json(params.dry_run, "dry-run", json);
    param_from_json(params.num_cells_per_rank, "num-cells-per-rank", json);
    param_from_json(params.num_ranks, "num-ranks", json);
    param_from_json(params.duration, "duration", json);
    param_from_json(params.min_delay, "min-delay", json);
    params.cell = parse_cell_parameters(json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }

    return params;
}
