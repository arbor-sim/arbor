#pragma once

#include <vector>

#include <communication/gathered_vector.hpp>
#include <communication/mpi.hpp>
#include <spike.hpp>

namespace arb {

struct mpi_context {
    int size_;
    int rank_;
    MPI_Comm  comm_;

    // throws std::runtime_error if MPI calls fail
    mpi_context(MPI_Comm comm=MPI_COMM_WORLD): comm_(comm) {
        size_ = arb::mpi::size(comm_);
        rank_ = arb::mpi::rank(comm_);
    }

    gathered_vector<arb::spike>
    gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        return mpi::gather_all_with_partition(local_spikes, comm_);
    }

    int id() const {
        return rank_;
    }

    int size() const {
        return size_;
    }

    template <typename T>
    T min(T value) const {
        return arb::mpi::reduce(value, MPI_MIN, comm_);
    }

    template <typename T>
    T max(T value) const {
        return arb::mpi::reduce(value, MPI_MAX, comm_);
    }

    template <typename T>
    T sum(T value) const {
        return arb::mpi::reduce(value, MPI_SUM, comm_);
    }

    template <typename T>
    std::vector<T> gather(T value, int root) const {
        return mpi::gather(value, root, comm_);
    }

    void barrier() const {
        mpi::barrier(comm_);
    }

    std::string name() const {
        return "MPI";
    }
};

} // namespace arb

