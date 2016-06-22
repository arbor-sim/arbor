#pragma once

#include <type_traits>
#include <vector>

#include <cstdint>

#include <communication/spike.hpp>

namespace nest {
namespace mc {
namespace communication {

struct serial_global_policy {
    std::vector<spike<uint32_t>> const
    gather_spikes(const std::vector<spike<uint32_t>>& local_spikes) {
        return local_spikes;
    }

    static int id() {
        return 0;
    }

    static int num_communicators() {
        return 1;
    }

    template <typename T>
    T min(T value) const {
        return value;
    }

    template <typename T>
    T max(T value) const {
        return value;
    }

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value>
    >
    std::vector<T> make_map(T local) {
        return {T(0), local};
    }
};

} // namespace communication
} // namespace mc
} // namespace nest
