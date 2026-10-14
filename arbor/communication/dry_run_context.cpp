#include <string>
#include <vector>

#include <arbor/spike.hpp>

#include "distributed_context.hpp"
#include "label_resolution.hpp"
#include "util/rangeutil.hpp"

namespace arb {

// Only rank 0 is simulated. Every other rank r holds a copy of rank 0's tile of
// cells, with gids shifted by r tiles, modulo the number of cells (see
// symmetric_recipe). By this symmetry, whatever rank r sends to rank 0 is what
// rank 0 sends to rank -r, shifted by r tiles.
struct dry_run_tiling {
    unsigned num_ranks;
    unsigned num_cells_per_tile;

    // Shift gid from rank 0's view to rank's.
    cell_gid_type shift(cell_gid_type gid, unsigned rank) const {
        return (gid + rank*num_cells_per_tile)%(num_ranks*num_cells_per_tile);
    }

    // Rank r sends to rank 0 what rank 0 sends to rank -r.
    unsigned mirror(unsigned rank) const {
        return (num_ranks - rank)%num_ranks;
    }
};

struct dry_run_context_impl {
    using count_type = typename gathered_vector<spike>::count_type;

    explicit dry_run_context_impl(unsigned num_ranks, unsigned num_cells_per_tile):
        num_ranks_(num_ranks), num_cells_per_tile_(num_cells_per_tile) {};
    std::vector<spike>
    remote_gather_spikes(const std::vector<spike>& local_spikes) const {
        return {};
    }
    dry_run_tiling tiling() const { return {num_ranks_, num_cells_per_tile_}; }

    gathered_vector<spike>
    all_to_all_spikes(const gathered_vector<spike>& local_spikes) const {
        const auto& local = local_spikes.values();
        const auto& local_part = local_spikes.partition();
        std::vector<spike> gathered_spikes;
        gathered_spikes.reserve(static_cast<size_t>(local_spikes.size()));
        std::vector<count_type> partition;
        partition.reserve(num_ranks_ + 1);
        partition.push_back(0);
        for (unsigned ridx = 0; ridx < num_ranks_; ridx++) {
            auto from = tiling().mirror(ridx);
            for (auto lidx = local_part[from]; lidx < local_part[from + 1]; ++lidx) {
                auto spike = local[lidx];
                spike.source.gid = tiling().shift(spike.source.gid, ridx);
                gathered_spikes.push_back(spike);
            }
            partition.push_back(gathered_spikes.size());
        }
        return gathered_vector<spike>(std::move(gathered_spikes), std::move(partition));
    }
    gathered_vector<spike>
    gather_spikes(const std::vector<spike>& local_spikes) const {

        count_type local_size = local_spikes.size();

        std::vector<spike> gathered_spikes;
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

        return gathered_vector<spike>(std::move(gathered_spikes), std::move(partition));
    }
    void remote_ctrl_send_continue(const epoch&) const {}
    void remote_ctrl_send_done() const {}
    gathered_vector<cell_gid_type>
    gather_gids(const std::vector<cell_gid_type>& local_gids) const {
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
            partition.push_back(i*local_size);
        }

        return gathered_vector<cell_gid_type>(std::move(gathered_gids), std::move(partition));
    }

    distributed_neighborhood_ptr make_neighborhood(const std::vector<int>&) const {
        throw arbor_internal_error("make_neighborhood: not implemented for dry run context.");
    }

    // Rank r needs from rank 0 what rank 0 needs from rank -r.
    gathered_vector<cell_member_type>
    all_to_all_gids_domains(const std::vector<std::vector<cell_member_type>>& gids_domains) const {
        using count_type = gathered_vector<cell_member_type>::count_type;
        std::vector<cell_member_type> gathered_gids;
        std::vector<count_type> partition{0};
        for (unsigned ridx = 0; ridx < num_ranks_; ridx++) {
            for (auto src: gids_domains[tiling().mirror(ridx)]) {
                src.gid = tiling().shift(src.gid, ridx);
                gathered_gids.push_back(src);
            }
            partition.push_back(gathered_gids.size());
        }
        return gathered_vector<cell_member_type>(std::move(gathered_gids), std::move(partition));
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

    std::vector<std::size_t> gather_all(std::size_t value) const {
        return std::vector<std::size_t>(num_ranks_, value);
    }

    distributed_request send_recv_nonblocking(std::size_t dest_count,
        void* dest_data,
        int dest,
        std::size_t source_count,
        const void* source_data,
        int source,
        int tag) const {
        throw arbor_internal_error("send_recv_nonblocking: not implemented for dry run conext.");

        return distributed_request{
            std::make_unique<distributed_request::distributed_request_interface>()};
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
