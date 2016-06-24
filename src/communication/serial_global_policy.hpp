#pragma once

#include <type_traits>
#include <vector>

#include <cstdint>

#include <communication/spike.hpp>

namespace nest {
namespace mc {
namespace communication {

struct serial_global_policy {
    using id_type = uint32_t;

    std::vector<spike<id_type>> const
    gather_spikes(const std::vector<spike<id_type>>& local_spikes) {
        return local_spikes;
    }

    static int id() {
        return 0;
    }

    static int size() {
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
