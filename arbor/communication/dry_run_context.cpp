#include <algorithm>
#include <string>
#include <vector>

#include <arbor/spike.hpp>

#include "distributed_context.hpp"
#include "label_resolution.hpp"
#include "threading/threading.hpp"
#include "util/rangeutil.hpp"

namespace arb {

struct dry_run_context_impl {

    explicit dry_run_context_impl(unsigned num_ranks, unsigned num_cells_per_tile):
        num_ranks_(num_ranks), num_cells_per_tile_(num_cells_per_tile) {};

    gathered_vector<arb::spike>
    gather_spikes(const std::vector<arb::spike>& local_spikes) const {
        using count_type = typename gathered_vector<arb::spike>::count_type;

        count_type local_size = local_spikes.size();

        std::vector<arb::spike> gathered_spikes;
        gathered_spikes.reserve(local_size*num_ranks_);

        for (count_type i = 0; i < num_ranks_; i++) {
            util::append(gathered_spikes, local_spikes);
        }

        for (count_type i = 0; i < num_ranks_; i++) {
            for (count_type j = i*local_size; j < (i+1)*local_size; j++){
                gathered_spikes[j].source.gid += num_cells_per_tile_*i;
            }
        }

        std::vector<count_type> partition;
        for (count_type i = 0; i <= num_ranks_; i++) {
            partition.push_back(static_cast<count_type>(i*local_size));
        }

        return gathered_vector<arb::spike>(std::move(gathered_spikes), std::move(partition));
    }

    gathered_vector<cell_gid_type>
    gather_gids(const std::vector<cell_gid_type>& local_gids) const {
        using count_type = typename gathered_vector<cell_gid_type>::count_type;

        count_type local_size = local_gids.size();

        std::vector<cell_gid_type> gathered_gids;
        gathered_gids.reserve(local_size*num_ranks_);

        for (count_type i = 0; i < num_ranks_; i++) {
            util::append(gathered_gids, local_gids);
        }

        for (count_type i = 0; i < num_ranks_; i++) {
            for (count_type j = i*local_size; j < (i+1)*local_size; j++){
                gathered_gids[j] += num_cells_per_tile_*i;
            }
        }

        std::vector<count_type> partition;
        for (count_type i = 0; i <= num_ranks_; i++) {
            partition.push_back(static_cast<count_type>(i*local_size));
        }

        return gathered_vector<cell_gid_type>(std::move(gathered_gids), std::move(partition));
    }

    std::vector<std::vector<cell_gid_type>>
    gather_gj_connections(const std::vector<std::vector<cell_gid_type>> & local_connections) const {
        auto local_size = local_connections.size();
        std::vector<std::vector<cell_gid_type>> global_connections;
        global_connections.reserve(local_size*num_ranks_);

        for (unsigned i = 0; i < num_ranks_; i++) {
            util::append(global_connections, local_connections);
        }

        for (unsigned i = 0; i < num_ranks_; i++) {
            for (unsigned j = i*local_size; j < (i+1)*local_size; j++){
                for (auto& conn_gid: global_connections[j]) {
                    conn_gid += num_cells_per_tile_*i;
                }
            }
        }
        return global_connections;
    }

    cell_label_range gather_cell_label_range(const cell_label_range& local_ranges) const {
        cell_label_range global_ranges;
        for (unsigned i = 0; i < num_ranks_; i++) {
            global_ranges.append(local_ranges);
        }
        return global_ranges;
    }

    cell_labels_and_gids gather_cell_labels_and_gids(const cell_labels_and_gids& local_labels_and_gids) const {
        auto global_ranges = gather_cell_label_range(local_labels_and_gids.label_range);
        auto gids = gather_gids(local_labels_and_gids.gids);
        return cell_labels_and_gids(global_ranges, gids.values());
    }

    template <typename T>
    std::vector<T> gather(T value, int) const {
        return std::vector<T>(num_ranks_, value);
    }

    int id() const { return 0; }

    int size() const { return num_ranks_; }

    template <typename T>
    T min(T value) const { return value; }

    template <typename T>
    T max(T value) const { return value; }

    template <typename T>
    T sum(T value) const { return value * num_ranks_; }

    void barrier() const {}

    std::string name() const { return "dryrun"; }

    unsigned num_ranks_;
    unsigned num_cells_per_tile_;
};

ARB_ARBOR_API std::shared_ptr<distributed_context> make_dry_run_context(unsigned num_ranks, unsigned num_cells_per_tile) {
    return std::make_shared<distributed_context>(dry_run_context_impl(num_ranks, num_cells_per_tile));
}

} // namespace arb
