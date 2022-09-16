#include <iostream>

#include <arbor/context.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/version.hpp>

#ifndef ARB_MPI_ENABLED
#error "This is an MPI only example. No MPI found."
#endif

#include <mpi.h>

using arb::util::unique_any;
using arb::cell_size_type;
using arb::cell_tag_type;
using arb::cell_gid_type;

struct recipe: public arb::recipe {
        cell_size_type size = 16;
        float weight = 0.5;
        float delay = 1.0;

        recipe() {}
        cell_size_type num_cells() const override { return size; }
        unique_any get_cell_description(cell_gid_type) const override {
            auto lif = arb::lif_cell("src", "tgt");
            lif.tau_m   = 10;
            lif.V_th    = 10;
            lif.C_m     = 20;
            lif.E_L     = 0;
            lif.V_m     = 0;
            lif.V_reset = 0;
            lif.t_ref   = 2;
            return lif;
        }
        std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
            if (gid == 0) {
                return {arb::regular_generator({"tgt"}, weight, 1, 0.25)};
            }
            else {
                return {};
            }
        }
        std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
            return {arb::cell_connection{{ (gid-1) % size, arb::cell_local_label_type{"src"}}, {"tgt"}, weight, delay}};
        }
        arb::cell_kind get_cell_kind(cell_gid_type) const override { return arb::cell_kind::lif; }
};

// Stolen from Arbor's mpi internals
template <typename T>
std::vector<arb::spike> gather_all(const std::vector<arb::spike>& values, MPI_Comm comm) {
    int size = 0;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(0, size), displs(0, size);

    counts = gather_all(int(values.size()), comm);
    for (auto& c : counts) {
        c *= traits::count();
    }
    util::make_partition(displs, counts);

    std::vector<T> buffer(displs.back()/traits::count());
    MPI_OR_THROW(MPI_Allgatherv,
            // const_cast required for MPI implementations that don't use const* in their interfaces
            const_cast<T*>(values.data()), counts[rank(comm)], traits::mpi_type(),  // send buffer
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(), // receive buffer
            comm);

    return buffer;
}


int main() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    int rank = -1, size = -1;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (size < 2) {
        std::cerr << "We need to split the communicator into two; thus please launch with >= 2 tasks.\n";
        return -1;
    }
    // Splitting predicate; we put the root into the first group
    auto group = rank == 0;
    // Set up and intra-comm splitting the root comm in two
    MPI_Comm split;
    MPI_Comm_split(comm, group, rank, &split);
    int lrank, lsize;
    MPI_Comm_size(comm, &lsize);
    MPI_Comm_rank(comm, &lrank);
    // Now make an inter comm between the two
    MPI_Comm inter;
    // Figure out what our leader's rank is in WORLD
    // ideally one would elect this by checking `lrank == 0` and broadcasting
    // that tasks `rank` over `split`. However, this is not an excercise in MPI
    // and we use this ceremony just to get an intercomm that will later be made
    // with MPI_Comm_connect/_accept.
    int peer_lead = (group == 0) ? 0 : 1;
    MPI_Intercomm_create(split, 0,         // local comm + local leader rank
                         comm, peer_lead,  // peer  comm + peer  leader rank
                         42,               // tag ... no idea what it does.
                         &inter);          // new intercomm
    if (group == 1) {
        // Run the simulation for slightly more than one epoch; thus we expect a single
        // inter-communcation to happend.
        auto ctx = arb::make_context({}, split);
        auto rec = recipe(); // one cell per process
        auto sim = arb::simulation(rec, ctx);
        sim.connect_to_remote_simulation(std::any{inter});
        sim.run(1.01, 0.025);
    }
    else {

    }
    MPI_Finalize();
}
