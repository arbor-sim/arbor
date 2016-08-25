#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include <communication/gathered_vector.hpp>
#include <spike.hpp>

namespace nest {
namespace mc {
namespace communication {

struct serial_global_policy {
    template <typename Spike>
    static gathered_vector<Spike>
    gather_spikes(const std::vector<Spike>& local_spikes) {
        return gathered_vector<Spike>(std::vector<Spike>(local_spikes), {0u, 1u});
    }

    static int id() {
        return 0;
    }

    static int size() {
        return 1;
    }

    template <typename T>
    static T min(T value) {
        return value;
    }

    template <typename T>
    static T max(T value) {
        return value;
    }

    template <typename T>
    static T sum(T value) {
        return value;
    }

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value>
    >
    static std::vector<T> make_map(T local) {
        return {T(0), local};
    }

    static void setup(int& argc, char**& argv) {}
    static void teardown() {}
    static const char* name() { return "serial"; }
};

} // namespace communication
} // namespace mc
} // namespace nest
