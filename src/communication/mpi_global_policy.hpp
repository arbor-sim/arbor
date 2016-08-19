#pragma once

#ifndef WITH_MPI
#error "mpi_global_policy.hpp should only be compiled in a WITH_MPI build"
#endif

#include <cstdint>
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
        return mpi::gather_all_meta(local_spikes);
    }

    static int id() { return mpi::rank(); }

    static int size() { return mpi::size(); }

    template <typename T>
    static T min(T value) {
        return nest::mc::mpi::reduce(value, MPI_MIN);
    }

    template <typename T>
    static T max(T value) {
        return nest::mc::mpi::reduce(value, MPI_MAX);
    }

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value>
    >
    static std::vector<T> make_map(T local) {
        return algorithms::make_index(mpi::gather_all(local));
    }

    static void setup(int& argc, char**& argv) {
        nest::mc::mpi::init(&argc, &argv);
    }

    static void teardown() {
        nest::mc::mpi::finalize();
    }

    static const char* name() { return "MPI"; }

private:
};

} // namespace communication
} // namespace mc
} // namespace nest

