#pragma once

#include <type_traits>
#include <vector>

#include <cstdint>

#include <communication/spike.hpp>
#include <algorithms.hpp>

#include "mpi.hpp"

namespace nest {
namespace mc {
namespace communication {

struct mpi_global_policy {
    std::vector<spike<uint32_t>> const
    gather_spikes(const std::vector<spike<uint32_t>>& local_spikes) {
        return mpi::gather_all(local_spikes);
    }

    int id() const {
        return mpi::rank();
    }

    /*
    template <typename T>
    T min(T value) const {
    }
    */

    int num_communicators() const {
        return mpi::size();
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
