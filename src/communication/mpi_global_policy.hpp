#pragma once

#ifndef NMC_HAVE_MPI
#error "mpi_global_policy.hpp should only be compiled in a NMC_HAVE_MPI build"
#endif

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <algorithms.hpp>
#include <common_types.hpp>
#include <communication/gathered_vector.hpp>
#include <communication/mpi.hpp>
#include <spike.hpp>

namespace nest {
namespace mc {
namespace communication {

struct mpi_global_policy {
    template <typename Spike>
    static gathered_vector<Spike>
    gather_spikes(const std::vector<Spike>& local_spikes) {
        return mpi::gather_all_with_partition(local_spikes);
    }

    static int id() { return mpi::rank(); }

    static int size() { return mpi::size(); }

    static void set_sizes(int comm_size, int num_local_cells) {
        throw std::runtime_error(
            "Attempt to set comm size for MPI global communication "
            "policy, this is only permitted for dry run mode");
    }

    template <typename T>
    static T min(T value) {
        return nest::mc::mpi::reduce(value, MPI_MIN);
    }

    template <typename T>
    static T max(T value) {
        return nest::mc::mpi::reduce(value, MPI_MAX);
    }

    template <typename T>
    static T sum(T value) {
        return nest::mc::mpi::reduce(value, MPI_SUM);
    }

    static void setup(int& argc, char**& argv) {
        nest::mc::mpi::init(&argc, &argv);
    }

    static void teardown() {
        nest::mc::mpi::finalize();
    }

    static global_policy_kind kind() { return global_policy_kind::mpi; };
};

using global_policy = mpi_global_policy;

} // namespace communication
} // namespace mc
} // namespace nest

