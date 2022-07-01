#include <any>
#include <iostream>
#include <unordered_map>

#include <arborio/label_parse.hpp>

#include <arbor/common_types.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
void sync() { MPI_Barrier(MPI_COMM_WORLD); }
#else
void sync() {}
#endif

using namespace arborio::literals;

// Fan out network with n members
// - one spike source at gid 0
// - n-1 passive, soma-only cable cells
// Starts out disconnected, but new connections from the source to the cable cells may be added.
struct recipe: public arb::recipe {
    const std::string
        syn = "synapse",  // handle for exponential synapse on cable cells
        det = "detector", //            spike detector on cable cells
        src = "src";      //            spike source
    const double
        r = 3.0,    // soma radius
        f = 0.0125, // spike source interval
        w = 0.75,   // connection weight
        d = 0.1;    //            delay
    arb::cell_size_type n_ = 0; // cell count
    mutable std::unordered_map<arb::cell_gid_type, std::vector<arb::cell_connection>> connected; // lookup table for connections

    // Required but uninteresting methods
    recipe(arb::cell_size_type n): n_{n} {}
    arb::cell_size_type num_cells() const override { return n_; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override { return 0 == gid ? arb::cell_kind::spike_source : arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind kind) const override {
        if (kind == arb::cell_kind::spike_source) return {};
        arb::cable_cell_global_properties ccp;
        ccp.default_parameters = arb::neuron_parameter_defaults;
        return ccp;
    }

    // Look up the (potential) connection to this cell
    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override { return connected[gid]; }

    // Connect cell `to` to the spike source
    void add_connection(arb::cell_gid_type to) { assert(to > 0); connected[to] = {arb::cell_connection({0, src}, {syn}, w, d)}; }

    // Return the cell at gid
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        // source at gid 0
        if (gid == 0) return arb::spike_source_cell{src, arb::regular_schedule(f)};
        // all others are receiving cable cells. Spike once after incoming spike and 'never' again.
        arb::segment_tree tree;
        tree.append(arb::mnpos, { -r, 0, 0, r}, {r, 0, 0, r}, 1);
        auto decor = arb::decor{};
        decor.paint("(all)"_reg, arb::density("pas"));
        decor.place("(location 0 0.5)"_ls, arb::synapse("expsyn"), syn);
        decor.place("(location 0 0.5)"_ls, arb::threshold_detector{-10.0}, det);
        return arb::cable_cell({tree}, {}, decor);
    }
};

// Prints the locally collected spikes -- rank by rank -- and then clears the buffer
void show_spikes(int ep, std::vector<arb::spike>& spikes, int rnk, int csz) {
    sync();
    if (!rnk) std::cout << "Epoch " << ep << '\n';
    for (int ix = 0; ix < csz; ++ix) {
        sync();
        if (ix == rnk) {
            std::cout << " * Rank " << rnk << '\n';
            for(const auto& spike: spikes) std::cout << "   * " << rnk << '/' << csz << ": " << spike.source << "@" << spike.time << '\n';
        }
        sync();
    }
    spikes.clear();
    sync();
}

int main(int argc, char** argv) {
#ifdef ARB_MPI_ENABLED
    arbenv::with_mpi guard(argc, argv, false);
    auto ctx = arb::make_context({}, MPI_COMM_WORLD);
#else
    auto ctx = arb::make_context();
#endif
    auto rnk = arb::rank(ctx);
    auto csz = arb::num_ranks(ctx);
    auto rec = recipe(csz);
    auto dec = arb::domain_decomposition(rec, ctx, {{rnk == 0 ? arb::cell_kind::spike_source : arb::cell_kind::cable, {rnk}, arb::backend_kind::multicore}});
    auto sim = arb::simulation(rec, ctx, dec);

    // Record spikes for each rank individually
    std::vector<arb::spike> spikes;
    sim.set_local_spike_callback([&spikes](const auto& s) { spikes.insert(spikes.end(), s.begin(), s.end()); });

    for (int ep = 1; ep < csz; ++ep) {
        // Add another connection from the spike source
        rec.add_connection(ep);
        sim.update_connections(rec);
        // Run for 0.25ms and print spikes
        sim.run(0.25*ep, 0.025);
        show_spikes(ep, spikes, rnk, csz);
    }
}
