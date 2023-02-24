#include <iomanip>
#include <iostream>
#include <cassert>

#include <arbor/context.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/version.hpp>

#ifndef ARB_MPI_ENABLED
#error "This is an MPI only example. No MPI found."
#endif

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

std::vector<arb::spike> gather_all(const std::vector<arb::spike>& spikes, MPI_Comm comm);

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
        std::cerr << "[EXT] Got min delay=" << mid << '\n';
        sim.add_sampler(arb::all_probes,
                        arb::regular_schedule(0.05),
                        sampler);
        sim.run(T, dt);
        std::cout << std::fixed << std::setprecision(4);
        for (const auto& [t, v]: trace) std::cout << t << " " << v << '\n';
        exit(0); // Force quit, since the sender doesn't know when to stop.
    }
    else if (mpi.group == 0) {
        // Simulate external spikes
        // * We sent from a fictious cell with gid=local rank
        // * The time is a function of rank and epoch.
        // * We never stop until the other group kills the process ;)
        for (int ep = 0;; ++ep) {
            arb::cell_gid_type src = mpi.local_rank;
            auto time = ep*2.0*dt + mpi.local_rank*dt + 0.05;
            std::vector<arb::spike> spikes(10, {{src, 0}, time});
            auto from_arbor = gather_all(spikes, mpi.inter);
            std::cerr << "[" << ep << "] spikes from Arbor: " << from_arbor.size() << '\n';
        }
    }
    else {
        // unreachable
    }
    MPI_Finalize();
}

std::vector<arb::spike> gather_all(const std::vector<arb::spike>& spikes, MPI_Comm comm) {
    int size = -1, rank = -1;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    int spike_count = spikes.size();
    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    MPI_Allgather(&spike_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int recv_bytes = 0;
    int recv_count = 0;
    for (int ix = 0; ix < size; ++ix) {
        recv_count += counts[ix];
        counts[ix] *= sizeof(arb::spike); // Number of B for rank `ix`
        displs[ix]  = recv_bytes;         // Offset for rank `ix` in B
        recv_bytes += counts[ix];         // Total number of items so far

    }
    std::vector<arb::spike> recv_spikes(recv_count);
    auto send_bytes = spikes.size()*sizeof(arb::spike);
    MPI_Allgatherv(spikes.data(),      send_bytes,                   MPI_BYTE, // send buffer
                   recv_spikes.data(), counts.data(), displs.data(), MPI_BYTE, // recv buffer
                   comm);
    return recv_spikes;
}

// MPI Ceremonies
mpi_handle setup_mpi() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm comm; MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    int rank = -1, size = -1;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (size < 2) {
        std::cerr << "We need to split the communicator into two; thus please launch with >=2 tasks.\n";
        exit(-1);
    }
    // Splitting predicate;
    // - even ranks to group=0, which will fill for an external process
    // - thus root goes to group=0
    // - odd ranks will run Arbor
    auto group = 0 == rank % 2;
    // Set up and intra-comm splitting the root comm in two
    MPI_Comm split; MPI_Comm_split(comm, group, rank, &split);
    int lrank, lsize;
    MPI_Comm_size(comm, &lsize);
    MPI_Comm_rank(comm, &lrank);
    // Figure out what our leader's rank is in WORLD
    // ideally one would elect this by checking `lrank == 0` and broadcasting
    // that task's `rank` over `split`. However, this is not an excercise in MPI
    // and we use this ceremony just to get an intercomm that will later be made
    // with MPI_Comm_connect/_accept.
    int peer_lead = (group == 0) ? 0 : 1;
    // Now make an inter comm between the two
    MPI_Comm inter;
    MPI_Intercomm_create(split, 0,        // local comm + local leader rank
                         comm, peer_lead, // peer  comm + peer  leader rank
                         42,              // tag, must match across call
                         &inter);         // new intercomm
    return {comm, rank, size, split, lrank, lsize, group, inter};
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
