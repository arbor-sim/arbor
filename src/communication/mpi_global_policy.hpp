#pragma once

#include <type_traits>
#include <vector>

#include <cstdint>

#include <communication/spike.hpp>
#include <communication/mpi.hpp>
#include <algorithms.hpp>


namespace nest {
namespace mc {
namespace communication {

struct mpi_global_policy {
    using id_type = uint32_t;

    std::vector<spike<id_type>> const
    gather_spikes(const std::vector<spike<id_type>>& local_spikes) {
        return mpi::gather_all(local_spikes);
    }

    int id() const { return mpi::rank(); }

    int size() const { return mpi::size(); }

    template <typename T>
    T min(T value) const {
        return nest::mc::mpi::reduce(value, MPI_MIN);
    }

    template <typename T>
    T max(T value) const {
        return nest::mc::mpi::reduce(value, MPI_MAX);
    }

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value>
    >
    std::vector<T> make_map(T local) {
        return algorithms::make_index(mpi::gather_all(local));
    }
};

} // namespace communication
} // namespace mc
} // namespace nest
