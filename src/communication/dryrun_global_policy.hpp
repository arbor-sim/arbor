#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include <communication/gathered_vector.hpp>
#include <util/span.hpp>
#include <spike.hpp>

namespace nest {
namespace mc {
namespace communication {

extern int dryrun_num_local_cells;
extern int dryrun_communicator_size;

struct dryrun_global_policy {
    template <typename Spike>
    static gathered_vector<Spike>
    gather_spikes(const std::vector<Spike>& local_spikes) {
        using util::make_span;
        using count_type = typename gathered_vector<Spike>::count_type;

        // Build the global spike list by replicating the local spikes for each
        // "dummy" domain.
        const auto num_spikes_local  = local_spikes.size();
        const auto num_spikes_global = size()*num_spikes_local;
        std::vector<Spike> global_spikes(num_spikes_global);
        std::vector<count_type> partition(size()+1);

        for (auto rank: make_span(0u, size())) {
            const auto first_cell = rank*dryrun_num_local_cells;
            const auto first_spike = rank*num_spikes_local;
            for (auto i: make_span(0, num_spikes_local)) {
                // the new global spike is the same as the local spike, with
                // its source index shifted to the dummy domain
                auto s = local_spikes[i];
                s.source.gid += first_cell;
                global_spikes[first_spike+i] = s;
            }
            partition[rank+1] = partition[rank]+num_spikes_local;
        }

        EXPECTS(partition.back()==num_spikes_global);
        return {std::move(global_spikes), std::move(partition)};
    }

    static int id() {
        return 0;
    }

    static int size() {
        return dryrun_communicator_size;
    }

    static void set_sizes(int comm_size, int num_local_cells) {
        dryrun_communicator_size = comm_size;
        dryrun_num_local_cells = num_local_cells;
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
        return size()*value;
    }

    template <typename T>
    static std::vector<T> gather(T value, int) {
        return std::vector<T>(size(), value);
    }

    static void barrier() {}

    static void setup(int& argc, char**& argv) {}
    static void teardown() {}

    static global_policy_kind kind() { return global_policy_kind::dryrun; };
};

using global_policy = dryrun_global_policy;

} // namespace communication
} // namespace mc
} // namespace nest
