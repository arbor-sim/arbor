#include <string>
#include <vector>

#include <arbor/spike.hpp>

#include "distributed_context.hpp"
#include "label_resolution.hpp"
#include "util/rangeutil.hpp"

namespace arb {

struct dry_run_context_impl {
    using count_type = typename gathered_vector<spike>::count_type;

    explicit dry_run_context_impl(unsigned num_ranks, unsigned num_cells_per_tile):
        num_ranks_(num_ranks), num_cells_per_tile_(num_cells_per_tile) {};
    std::vector<spike>
    remote_gather_spikes(const std::vector<spike>& local_spikes) const {
        return {};
    }

    // Generate a list of spikes for this rank (in dry run always zero) from
    // each other rank `i \in { 1 ... n-1}` (all ranks except zero). Given a
    // list of spikes (gid, lid, time), on this rank as a template and the
    // look-up structure from source to target (gid, lid) -> [rank]. Thus, we
    // need to shift the template list's `gid` to every tile, yielding one list
    // per rank. Then, we want to bin the resulting spikes using the source map.
    // However, we get the source map _for_ rank=0, i.e. the map of sources on
    // rank zero that we want to _ship to_ not _receive from_ rank=0. This means
    // we need to translate the source map _and_ the source gid.
    //
    // If, on rank=0, we had (gid, lid) -> [r0, r1, ...], and want to obtain the map on rank `r`
    // - we need to shift the gid -> (gid + r*cells-per-tile) % total-cells
    // - each rank `k` needs to be shifted by the current source rank
    gathered_vector<spike>
    all_to_all_spikes(const std::vector<spike>& spikes, const sources_to_target_ranks& lut) const {
        auto num_total_cells = num_cells_per_tile_*num_ranks_;
        std::vector<count_type> partition;
        partition.push_back(0);
        std::vector<spike> gathered_spikes;
        for (cell_gid_type rank = 0; rank < num_ranks_; ++rank) {
            for (const auto& spk: spikes) {
                // create a spike from a virtual rank by shifting the `gid` into
                // the `rank`th tile.
                auto shifted = spk.source;
                shifted.gid += num_cells_per_tile_*rank;
                shifted.gid %= num_total_cells;
                if (const auto to = lut.find_source(shifted); to != lut.end()) {
                    const auto& ranks = to->second;
                    // Search for the shifted rank.
                    // SAFETY: - ranks will never be empty by construction
                    //         - ranks is sorted by construction (via sort | uniq)
                    // auto it = std::find_if(ranks.begin(), ranks.end(),
                                           // [nr=num_ranks_, offset=num_ranks_ - rank](auto old) { return (old + offset) % nr == 0; }
                                          // );
                    auto it = std::find(ranks.begin(), ranks.end(), rank);
                    if (it != ranks.end()) gathered_spikes.emplace_back(shifted, spk.time);
                }
            }
            // NOTE there's no need to re-sort after shift/mod of the gid since
            //      shifting by a full tile will preserve the tile-internal ordering
            partition.push_back(gathered_spikes.size());
        }
        return gathered_vector<spike>(std::move(gathered_spikes), std::move(partition));
    }

    gathered_vector<spike>
    gather_spikes(const std::vector<spike>& local_spikes) const {
        auto num_cells = num_cells_per_tile_*num_ranks_;
        std::vector<spike> gathered_spikes;
        gathered_spikes.reserve(local_spikes.size()*num_ranks_);
        std::vector<count_type> partition;
        partition.push_back(0);
        for (count_type rank = 0; rank < num_ranks_; ++rank) {
            for (const auto& spk: local_spikes) {
                auto shifted = spk.source;
                shifted.gid += num_cells_per_tile_*rank;
                shifted.gid %= num_cells;
                gathered_spikes.emplace_back(shifted, spk.time);
            }
            // NOTE there's no need to re-sort after shift/mod of the gid since
            //      shifting by a full tile will preserve the tile-internal ordering
            partition.push_back(gathered_spikes.size());
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

    // The connections generated by symmetric_recipe (or any other method) will
    // produce an input vector to this function that looks conceptually like
    // this
    //
    // [ (gid on rank 0, lid), ... | (gid on rank 1, lid), ... | ... ]
    //
    // where the gid are interpreted as the _sources_ to connections terminating
    // on this rank. In dry run, 'this' is always zero.
    //
    // Our job in this function is to produce a list like this
    //
    // // [ (gid on rank 0, lid), ... | (gid on rank 0, lid), ... | ... ]
    //
    // which has the same layout, but different interpretation. The `i`th
    // segment in this vector is now supposed to be the list of sources
    // terminating on rank `j`.
    // 
    // We synthesise this by walking through the tiles in order
    // 
    // [ (gid on rank 0, lid), ... | (gid on rank 1, lid), ... | ... ]
    //
    // and working out the offset `off` to tile zero, ie 0, -1, -2, ... all mod
    // num_tiles. We shift all the gids in the tile by `off * num_cells_per_tile`
    //
    // [ (gid on rank 0, lid), ... | (gid on rank 1 - num_cells_per_tile, lid), ... | ... ]
    //
    // where gid on rank 1 - num_cells_per_tile ~ gid on rank 0 due to the
    // tiling property
    gathered_vector<cell_member_type>
    all_to_all_gids_domains(const std::vector<std::vector<cell_member_type>>& gids_domains) const {
        using count_type = gathered_vector<cell_member_type>::count_type;
        std::vector<count_type> partition(num_ranks_ + 1);
        partition[0] = 0;
        std::vector<cell_member_type> gathered_gids;
        for (count_type rank = 0; rank < num_ranks_; ++rank) {
            const auto from = (num_ranks_ - rank) % num_ranks_;
            const auto& chunk = gids_domains.at(from);
            for (const auto& src: chunk) {
                cell_member_type tmp = src;
                tmp.gid -= num_cells_per_tile_*from; // or % num_cells_per_tile
                gathered_gids.push_back(tmp); 
            }
            partition[rank + 1] = gathered_gids.size();            
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
