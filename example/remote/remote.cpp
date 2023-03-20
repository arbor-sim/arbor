#include <iomanip>
#include <iostream>
#include <cassert>

#include <arbor/context.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/version.hpp>

#include <mpi.h>
#include <arbor/communication/remote.hpp>
#include <arborenv/with_mpi.hpp>

struct remote_recipe: public arb::recipe {
        arb::cell_size_type size = 1;
        float weight = 20.0f;
        float delay = 0.1f;

        arb::cell_size_type num_cells() const override { return size; }
        arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
            auto lif = arb::lif_cell("src", "tgt");
            lif.tau_m =   2.0;
            lif.V_th  = -10.0;
            lif.C_m   =  20.0;
            lif.E_L   = -23.0;
            lif.V_m   = -23.0;
            lif.E_R   = -23.0;
            lif.t_ref =   0.2;
            return lif;
        }
        std::vector<arb::ext_cell_connection> external_connections_on(arb::cell_gid_type) const override {
            std::vector<arb::ext_cell_connection> res;
            // Invent some fictious cells outside of Arbor's realm
            for (arb::cell_gid_type gid = 0; gid < 10; gid++) {
                res.emplace_back(arb::cell_remote_label_type{gid}, arb::cell_local_label_type{"tgt"}, weight, delay);
            }
            return res;
        }
        arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::lif; }
        std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override { return {arb::lif_probe_voltage{}}; }
};

struct mpi_handle {
    MPI_Comm global = MPI_COMM_NULL;      // Global communicator; all processes live here
    int rank = -1, size = -1;             // Global coordinates
    MPI_Comm local = MPI_COMM_NULL;       // Local communicator; processes of the group live here
    int local_rank = -1, local_size = -1; // Local coordinates
    int group = -1;                       // Group index, in {0, 1}
    MPI_Comm inter = MPI_COMM_NULL;       // Intercommunicator between the groups
};

mpi_handle setup_mpi();

void sampler(arb::probe_metadata, std::size_t, const arb::sample_record*);

std::vector<std::pair<double, double>> trace;

int main() {
    // Setup
    auto T = 1.0;
    auto dt = 0.025;
    auto mpi = setup_mpi();
    if (mpi.group == 1) {
        auto ctx = arb::make_context({}, mpi.local, mpi.inter);
        auto rec = remote_recipe();
        auto sim = arb::simulation(rec, ctx);
        double mid = sim.min_delay();
        std::cerr << "[ARB]" << mpi.rank << " " << mpi.local_rank << '\n';
        std::cerr << "[ARB] Got min delay=" << mid << '\n';
        sim.add_sampler(arb::all_probes,
                        arb::regular_schedule(0.05),
                        sampler);
        sim.run(T, dt);
        std::cout << std::fixed << std::setprecision(4);
        std::cerr << "[ARB] Trace\n";
        for (const auto& [t, v]: trace) std::cout << " " << t << " " << v << '\n';
    }
    else if (mpi.group == 0) {
        // Simulate external spikes
        // * We send from a fictious cell with gid=local rank
        // * The time is a function of rank and epoch.
        // * We never stop until the other group kills the process ;)
        std::cerr << "[EXT]" << mpi.rank << " " << mpi.local_rank << '\n';
        for (int ep = 0;; ++ep) {
            std::cerr << "[EXT] Epoch " << ep << '\n';
            arb::cell_gid_type src = mpi.local_rank;
            auto time = ep*2.0*dt + mpi.local_rank*dt + 0.05;
            std::vector<arb::remote::arb_spike> spikes(10, {{src, 0}, time});
            std::cerr << "[EXT] Waiting for control message from Arbor\n";
            auto msg = arb::remote::exchange_ctrl(arb::remote::msg_epoch{}, mpi.inter);
            if(!std::holds_alternative<arb::remote::msg_epoch>(msg)) break;
            auto mep = std::get<arb::remote::msg_epoch>(msg);
            std::cerr << "[EXT] Waiting for spikes from Arbor for epoch [" << mep.t_start << ", " << mep.t_end << ")\n";
            auto from_arbor = arb::remote::gather_spikes(spikes, mpi.inter);
            std::cerr << "[EXT] spikes from Arbor: " << from_arbor.size() << '\n';
        }
        std::cerr << "[EXT] Arbor asked to quit; EXIT\n";
    }
    else {
        // unreachable
    }
    MPI_Finalize();
}

// MPI Ceremonies
mpi_handle setup_mpi() {
    mpi_handle result;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_dup(MPI_COMM_WORLD, &result.global);
    MPI_Comm_size(result.global, &result.size);
    MPI_Comm_rank(result.global, &result.rank);
    if (result.size < 2) {
        std::cerr << "We need to split the communicator into two; thus please launch with >=2 tasks.\n";
        exit(-1);
    }
    // Splitting predicate;
    // - even ranks to group=0, which will fill for an external process
    // - thus root goes to group=0
    // - odd ranks will run Arbor
    result.group = 0 == result.rank % 2;
    // Set up and intra-comm splitting the root comm in two
    MPI_Comm_split(result.global, result.group, result.rank, &result.local);
    MPI_Comm_size(result.local, &result.local_size);
    MPI_Comm_rank(result.local, &result.local_rank);
    // Figure out what our leader's rank is in WORLD
    // ideally one would elect this by checking `lrank == 0` and broadcasting
    // that task's `rank` over `split`. However, this is not an excercise in MPI
    // and we use this ceremony just to get an intercomm that will later be made
    // with MPI_Comm_connect/_accept.
    int peer_lead = (result.group == 0) ? 0 : 1;
    // Now make an inter comm between the two
    MPI_Intercomm_create(result.local, 0,          // local comm + local leader rank
                         result.global, peer_lead, // peer  comm + peer  leader rank
                         42,                       // tag, must match across call
                         &result.inter);           // new intercomm
    return result;
}

void sampler(arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
    for (std::size_t ix = 0; ix < n; ++ix) {
        if (pm.id.gid != 0) continue;
        if (pm.id.index != 0) continue;
        if (pm.tag != 0) continue;
        const auto& sample = samples[ix];
        auto value = *arb::util::any_cast<double*>(sample.data);
        auto time  = sample.time;
        trace.emplace_back(time, value);
    }
}
