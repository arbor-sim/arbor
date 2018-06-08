#pragma once

#include <vector>

#include <communication/gathered_vector.hpp>
#include <spike.hpp>

namespace arb {

struct local_context {
    gathered_vector<arb::spike>
    gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        using count_type = typename gathered_vector<arb::spike>::count_type;
        return gathered_vector<arb::spike>(
            std::vector<arb::spike>(local_spikes),
            {0u, static_cast<count_type>(local_spikes.size())}
        );
    }

    int id() const {
        return 0;
    }

    int size() const {
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

    template <typename T>
    T sum(T value) const {
        return value;
    }

    template <typename T>
    std::vector<T> gather(T value, int) const {
        return {std::move(value)};
    }

    void barrier() const {}

    std::string name() const {
        return "local";
    }
};

} // namespace arb
