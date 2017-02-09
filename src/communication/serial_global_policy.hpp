#pragma once

#include <cstdint>
#include <stdexcept>
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
        using count_type = typename gathered_vector<Spike>::count_type;
        return gathered_vector<Spike>(
            std::vector<Spike>(local_spikes),
            {0u, static_cast<count_type>(local_spikes.size())}
        );
    }

    static int id() {
        return 0;
    }

    static int size() {
        return 1;
    }

    static void set_sizes(int comm_size, int num_local_cells) {
        throw std::runtime_error(
            "Attempt to set comm size for serial global communication "
            "policy, this is only permitted for dry run mode");
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

    static void setup(int& argc, char**& argv) {}
    static void teardown() {}

    static global_policy_kind kind() { return global_policy_kind::serial; };
};

using global_policy = serial_global_policy;

} // namespace communication
} // namespace mc
} // namespace nest
