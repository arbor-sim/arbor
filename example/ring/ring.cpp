/*
 * A miniapp that demonstrates how to make a ring model
 *
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/distributed_context.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>

#include "hardware/node_info.hpp"
#include "load_balance.hpp"

#include "json_meter.hpp"
#include "parameters.hpp"

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

// Writes voltage trace as a json file.
void write_trace_json(const arb::trace_data<double>& trace);

class ring_recipe: public arb::recipe {
public:
    ring_recipe(unsigned num_cells, cell_parameters params, unsigned min_delay):
        num_cells_(num_cells),
        cell_params_(params),
        min_delay_(min_delay)
    {}

    // There is just the one cell in the model
    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        return branch_cell(gid, cell_params_);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable1d_neuron;
    }

    // Each cell has one spike detector (at the soma)
    cell_size_type num_sources(cell_gid_type gid) const override {
        return 1;
    }

    // The cell has one target synapse, which will be connected to cell gid-1
    cell_size_type num_targets(cell_gid_type gid) const override {
        return 1;
    }

    // Each cell has one incoming connection, from cell with gid-1
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        cell_gid_type src = gid? gid-1: num_cells_-1;
        return {arb::cell_connection({src, 0}, {gid, 0}, event_weight_, min_delay_)};
    }

    // Return one event generator on gid 0. This generates a single event that will
    // kick start the spiking.
    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        std::vector<arb::event_generator> gens;
        if (!gid) {
            gens.push_back(arb::vector_backed_generator({0u, 0u}, event_weight_, {0.f}));
        }
        return gens;
    }

    // There is one probe (for measuring voltage at the soma) on the cell
    cell_size_type num_probes(cell_gid_type gid)  const override {
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        // Get the appropriate kind for measuring voltage
        cell_probe_address::probe_kind kind = cell_probe_address::membrane_voltage;
        // Measure at the soma
        arb::segment_location loc(0, 0.0);

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }

private:
    cell_size_type num_cells_;
    cell_parameters cell_params_;
    double min_delay_;
    float event_weight_ = 0.01;
};

struct cell_stats {
    using size_type = std::uint64_t;
    size_type ncells = 0;
    size_type nsegs = 0;
    size_type ncomp = 0;

    cell_stats(arb::distributed_context* ctx, arb::recipe& r) {
        size_type nranks = ctx->size();
        size_type rank = ctx->id();
        ncells = r.num_cells();
        size_type cells_per_rank = ncells/nranks;
        size_type b = rank*cells_per_rank;
        size_type e = (rank==nranks-1)? ncells: (rank+1)*cells_per_rank;
        for (size_type i=b; i<e; ++i) {
            auto c = arb::util::any_cast<arb::mc_cell>(r.get_cell_description(i));
            nsegs += c.num_segments();
            ncomp += c.num_compartments();
        }
        nsegs = ctx->sum(nsegs);
        ncomp = ctx->sum(ncomp);
    }

    friend std::ostream& operator<<(std::ostream& o, const cell_stats& s) {
        return o << "cell stats: "
                 << s.ncells << " cells; "
                 << s.nsegs << " segments; "
                 << s.ncomp << " compartments.";
    }
};


int main(int argc, char** argv) {
    // A distributed_context is required for distributed computation (e.g. MPI).
    arb::distributed_context context;

    try {
#ifdef ARB_MPI_ENABLED
        with_mpi guard(argc, argv, false);
        context = mpi_context(MPI_COMM_WORLD);
#endif

        auto params = read_options(argc, argv);

        arb::profile::meter_manager meters(&context);
        meters.start();

        // Create an instance of our recipe.
        ring_recipe recipe(params.num_cells, params.cell, params.min_delay);
        cell_stats stats(&context, recipe);
        std::cout << stats << "\n";

        // Make the domain decomposition for the model
        auto node = arb::hw::get_node_info();
        auto decomp = arb::partition_load_balance(recipe, node, &context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, &context);

        // Set up the probe that will measure voltage in the cell.

        // The id of the only probe on the cell: the cell_member type points to (cell 0, probe 0)
        auto probe_id = cell_member_type{0, 0};
        // The schedule for sampling is 10 samples every 1 ms.
        auto sched = arb::regular_schedule(0.1);
        // This is where the voltage samples will be stored as (time, value) pairs
        arb::trace_data<double> voltage;
        // Now attach the sampler at probe_id, with sampling schedule sched, writing to voltage
        sim.add_sampler(arb::one_probe(probe_id), sched, arb::make_simple_sampler(voltage));

        // Set up output of the global spike list by the root rank.
        std::ofstream fid;
        if (context.id()==0) {
            std::string fname = "spikes.gdf";

            fid.open(fname);
            if (!fid.good()) {
                std::cerr << "Warning: unable to open file " << fname << " for spike output\n";
            }
            else {
                sim.set_global_spike_callback(
                    [&fid](const std::vector<arb::spike>& spikes) {
                        for (auto spike : spikes) {
                            char linebuf[45];
                            auto n = std::snprintf(
                                linebuf, sizeof(linebuf), "%u %.4f\n",
                                unsigned{spike.source.gid}, float(spike.time));
                            fid.write(linebuf, n);
                        }
                    });
            }
        }

        meters.checkpoint("model-init");

        // Run the simulation for 100 ms, with time steps of 0.025 ms.
        sim.run(params.duration, 0.025);

        meters.checkpoint("model-run");

        auto ns = sim.num_spikes();
        std::cout << "\n" << ns << " spikes generated at rate of "
                  << params.duration/ns << " ms between spikes\n";

        // Write the samples to a json file.
        write_trace_json(voltage);

        auto report = arb::profile::make_meter_report(meters);
        std::cout << report;
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}

void write_trace_json(const arb::trace_data<double>& trace) {
    std::string path = "./voltages.json";

    nlohmann::json json;
    json["name"] = "ring demo";
    json["units"] = "mV";
    json["cell"] = "0.0";
    json["probe"] = "0";

    auto& jt = json["data"]["time"];
    auto& jy = json["data"]["voltage"];

    for (const auto& sample: trace) {
        jt.push_back(sample.t);
        jy.push_back(sample.v);
    }

    std::ofstream file(path);
    file << std::setw(1) << json << "\n";
}

