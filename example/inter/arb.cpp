/*
 * A miniapp that demonstrates using an external spike source.
 */

#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#include <aux/ioutil.hpp>
#include <aux/json_meter.hpp>
#include <aux/with_mpi.hpp>

#include <mpi.h>

#include "parameters.hpp"
#include "mpiutil.hpp"


using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

// Generate a cell.
arb::mc_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params);

class ring_recipe: public arb::recipe {
public:
    ring_recipe(unsigned num_cells, cell_parameters params, double min_delay, int num_nest_cells):
        num_cells_(num_cells),
        cell_params_(params),
        min_delay_(min_delay),
        num_nest_cells_(num_nest_cells)
    {}

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        return branch_cell(gid, cell_params_);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable1d_neuron;
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
        std::vector<arb::cell_connection> cons;
        cons.push_back({arb::cell_connection({src, 0}, {gid, 0}, event_weight_, min_delay_)});
        return cons;
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
        arb::segment_location loc(0, 0.0);

        return arb::probe_info{id, kind, cell_probe_address{loc, kind}};
    }

private:
    cell_size_type num_cells_;
    cell_parameters cell_params_;
    double min_delay_;
    float event_weight_ = 0.01;
    int num_nest_cells_;
};

struct cell_stats {
    using size_type = unsigned;
    size_type ncells = 0;
    size_type nsegs = 0;
    size_type ncomp = 0;

    cell_stats(arb::recipe& r, MPI_Comm comm) {
        int nranks, rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nranks);
        ncells = r.num_cells();
        size_type cells_per_rank = ncells/nranks;
        size_type b = rank*cells_per_rank;
        size_type e = (rank==nranks-1)? ncells: (rank+1)*cells_per_rank;
        size_type nsegs_tmp = 0;
        size_type ncomp_tmp = 0;
        for (size_type i=b; i<e; ++i) {
            auto c = arb::util::any_cast<arb::mc_cell>(r.get_cell_description(i));
            nsegs_tmp += c.num_segments();
            ncomp_tmp += c.num_compartments();
        }
        MPI_Allreduce(&nsegs_tmp, &nsegs, 1, MPI_UNSIGNED, MPI_SUM, comm);
        MPI_Allreduce(&ncomp_tmp, &ncomp, 1, MPI_UNSIGNED, MPI_SUM, comm);
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
        aux::with_mpi guard(argc, argv, false);

        auto info = get_comm_info(true);
        //bool root = info.global_rank == info.arbor_root;

        auto context = arb::make_context(arb::proc_allocation(), info.comm);

        //std::cout << aux::mask_stream(root);

        auto params = read_options(argc, argv);

        arb::profile::meter_manager meters;
        meters.start(context);

        on_local_rank_zero(info, [&] {
                std::cout << "ARB: starting handshake" << std::endl;
        });

        // hand shake #1: communicate cell populations
        broadcast((int)params.num_cells, MPI_COMM_WORLD, info.arbor_root);
        int num_nest_cells = broadcast(0,  MPI_COMM_WORLD, info.nest_root);

        on_local_rank_zero(info, [&] {
                std::cout << "ARB: num_nest_cells: " << num_nest_cells << std::endl;
        });

        // Create an instance of our recipe.
        ring_recipe recipe(params.num_cells, params.cell, params.min_delay, num_nest_cells);

        auto decomp = arb::partition_load_balance(recipe, context);

        // Construct the model.
        arb::simulation sim(recipe, decomp, context);

        // hand shake #2: min delay
        float arb_comm_time = sim.min_delay()/2;
        broadcast(arb_comm_time, MPI_COMM_WORLD, info.arbor_root);
        float nest_comm_time = broadcast(0.f, MPI_COMM_WORLD, info.nest_root);
        sim.min_delay(nest_comm_time*2);
        on_local_rank_zero(info, [&] {
                std::cout << "ARB: min_delay=" << sim.min_delay() << std::endl;
        });

        float delta = sim.min_delay()/2;
        float sim_duration = params.duration;
        unsigned steps = sim_duration/delta;
        if (steps*delta < sim_duration) ++steps;

        //hand shake #3: steps
        broadcast(steps, MPI_COMM_WORLD, info.arbor_root);

        // Set up recording of spikes to a vector on the root process.
        std::vector<arb::spike> recorded_spikes;
        on_local_rank_zero(info, [&] {
            sim.set_global_spike_callback(
                [&](const std::vector<arb::spike>& spikes) {
                    print_vec_comm("ARB", spikes);
                    recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                });
        });

        // Define the external spike source callback
        sim.set_external_spike_callback(extern_callback(info));

        meters.checkpoint("model-init", context);

        std::cout << "ARB: running simulation" << std::endl;
        // Run the simulation for 100 ms, with time steps of 0.025 ms.
        sim.run(params.duration, 0.025);

        meters.checkpoint("model-run", context);

        auto ns = sim.num_spikes();

        // Write spikes to file
        on_local_rank_zero(info, [&] {
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
        });

        //auto report = arb::profile::make_meter_report(meters, context);
        //std::cout << report;
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}

// Helper used to interpolate in branch_cell.
template <typename T>
double interp(const std::array<T,2>& r, unsigned i, unsigned n) {
    double p = i * 1./(n-1);
    double r0 = r[0];
    double r1 = r[1];
    return r[0] + p*(r1-r0);
}

arb::mc_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params) {
    arb::mc_cell cell;

    // Add soma.
    auto soma = cell.add_soma(12.6157/2.0); // For area of 500 μm².
    soma->rL = 100;
    soma->add_mechanism("hh");

    std::vector<std::vector<unsigned>> levels;
    levels.push_back({0});

    // Standard mersenne_twister_engine seeded with gid.
    std::mt19937 gen(gid);
    std::uniform_real_distribution<double> dis(0, 1);

    double dend_radius = 0.5; // Diameter of 1 μm for each cable.

    unsigned nsec = 1;
    for (unsigned i=0; i<params.max_depth; ++i) {
        // Branch prob at this level.
        double bp = interp(params.branch_probs, i, params.max_depth);
        // Length at this level.
        double l = interp(params.lengths, i, params.max_depth);
        // Number of compartments at this level.
        unsigned nc = std::round(interp(params.compartments, i, params.max_depth));

        std::vector<unsigned> sec_ids;
        for (unsigned sec: levels[i]) {
            for (unsigned j=0; j<2; ++j) {
                if (dis(gen)<bp) {
                    sec_ids.push_back(nsec++);
                    auto dend = cell.add_cable(sec, arb::section_kind::dendrite, dend_radius, dend_radius, l);
                    dend->set_compartments(nc);
                    dend->add_mechanism("pas");
                    dend->rL = 100;
                }
            }
        }
        if (sec_ids.empty()) {
            break;
        }
        levels.push_back(sec_ids);
    }

    // Add spike threshold detector at the soma.
    cell.add_detector({0,0}, 10);

    // Add a synapse to the mid point of the first dendrite.
    cell.add_synapse({1, 0.5}, "expsyn");

    return cell;
}

