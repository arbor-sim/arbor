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

#include <mpi.h>
#include <arborenv/with_mpi.hpp>

using namespace arborio::literals;

struct recipe: public arb::recipe {
    recipe() {
        ccp.default_parameters = arb::neuron_parameter_defaults;
    }

    arb::cell_size_type num_cells() const override { return 3; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override { return 0 == gid ? arb::cell_kind::spike_source : arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind kind) const override { return kind == arb::cell_kind::cable ? ccp : std::any{}; }

    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        if (gid == 0) {
            return arb::spike_source_cell{"source", arb::regular_schedule(0.0125)};
        } else {
            double r = 3;
            arb::segment_tree tree;
            tree.append(arb::mnpos, { -r, 0, 0, r}, {r, 0, 0, r}, 1);
            auto decor = arb::decor{};
            decor.paint("(all)"_reg, arb::density("pas"));
            decor.place("(location 0 0.5)"_ls, arb::synapse("expsyn"), "synapse");
            decor.place("(location 0 0.5)"_ls, arb::threshold_detector{-10.0}, "detector");
            return arb::cable_cell({tree}, {}, decor);
        }
    }

    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        if (connected.count(gid)) {
            const auto& [w, d] = connected.at(gid);
            return {arb::cell_connection({0, "source"}, {"synapse"}, w, d)};
        } else {
            return {};
        }
    }

    void add_connection(arb::cell_gid_type to) { connected[to] = {0.75, 0.1}; }

    std::unordered_map<arb::cell_gid_type, std::pair<double, double>> connected;
    arb::cable_cell_global_properties ccp;
};

void show_spikes(std::vector<arb::spike>& spikes, int rnk, int csz) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rnk) std::cout << "Epoch\n";
    for (int ix = 0; ix < csz; ++ix) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (ix == rnk) {
            std::cout << " * Rank " << rnk << '\n';
            for(const auto& spike: spikes) std::cout << "   * " << rnk << '/' << csz << ": " << spike.source << "@" << spike.time << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    spikes.clear();
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    arbenv::with_mpi guard(argc, argv, false);

    auto ctx = arb::make_context({}, MPI_COMM_WORLD);
    auto rnk = arb::rank(ctx);
    auto csz = arb::num_ranks(ctx);
    assert(csz == 3);

    auto rec = recipe();
    rec.add_connection(1);
    auto dec = arb::domain_decomposition(rec, ctx, {{rnk == 0 ? arb::cell_kind::spike_source : arb::cell_kind::cable, {rnk}, arb::backend_kind::multicore}});
    auto sim = arb::simulation(rec, ctx, dec);

    std::vector<arb::spike> spikes;
    if (rnk) sim.set_local_spike_callback([&spikes](const auto& s) { spikes.insert(spikes.end(), s.begin(), s.end()); });

    sim.run(0.25, 0.025);
    show_spikes(spikes, rnk, csz);

    rec.add_connection(2);
    sim.update_connections(rec, ctx, dec);

    sim.run(0.5, 0.025);
    show_spikes(spikes, rnk, csz);
}
