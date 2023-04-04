#include <iomanip>
#include <iostream>
#include <cassert>
#include <string>

#include <mpi.h>

#include <arbor/communication/remote.hpp>

struct mpi_handle {
    MPI_Comm comm = MPI_COMM_NULL;    // Local communicator; processes of the group live here
    int rank = -1, size = -1;         // Local coordinates
    MPI_Comm inter = MPI_COMM_NULL;   // Intercommunicator between the groups
    char secret[1024] = {0};          // Secret to pass to connect
};

// Pass in MPI_Comm_Accept Secret and get a communicator back.
mpi_handle setup_mpi();

int main() {
    // Setup
    auto dt = 0.025;
    auto mpi = setup_mpi();

    // Simulate external spikes
    // * We send from a fictious cell with gid=local rank
    // * The time is a function of rank and epoch.
    // * We never stop until the other group kills the process ;)
    std::cerr << "[EXT]" << mpi.rank << " " << mpi.rank << '\n';
    for (int ep = 0;; ++ep) {
        // Make some spikes
        std::cerr << "[EXT] Epoch " << ep << '\n';
        arb::remote::arb_gid_type src = mpi.rank;
        auto time = ep*2.0*dt + src*dt + 0.05;
        std::vector<arb::remote::arb_spike> spikes(10, {{src, 0}, time});

        // Send the control bit
        std::cerr << "[EXT] Waiting for control message from Arbor\n";
        auto msg = arb::remote::exchange_ctrl(arb::remote::msg_epoch{}, mpi.inter);
        if(!std::holds_alternative<arb::remote::msg_epoch>(msg)) break;
        auto mep = std::get<arb::remote::msg_epoch>(msg);

        // Exchange spikes
        // -> Send the local spikes on this rank
        // -> Receive all the spikes from Arbor
        std::cerr << "[EXT] Waiting for spikes from Arbor for epoch [" << mep.t_start << ", " << mep.t_end << ")\n";
        auto from_arbor = arb::remote::gather_spikes(spikes, mpi.inter);
        std::cerr << "[EXT] spikes from Arbor: " << from_arbor.size() << '\n';
    }
    std::cerr << "[EXT] Arbor asked to quit; EXIT\n";
    MPI_Finalize();
}

// MPI Ceremonies
mpi_handle setup_mpi() {
    mpi_handle result;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_dup(MPI_COMM_WORLD, &result.comm);
    MPI_Comm_size(result.comm, &result.size);
    MPI_Comm_rank(result.comm, &result.rank);

    MPI_Open_port(MPI_INFO_NULL, result.secret);
    std::cerr << result.secret << '\n';
    MPI_Comm_accept(result.secret, MPI_INFO_NULL, 0, result.comm, &result.inter);

    return result;
}
