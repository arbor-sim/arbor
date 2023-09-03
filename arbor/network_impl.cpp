#include "network_impl.hpp"
#include "cell_group_factory.hpp"
#include "communication/distributed_for_each.hpp"
#include "label_resolution.hpp"
#include "network_impl.hpp"
#include "threading/threading.hpp"
#include "util/rangeutil.hpp"
#include "util/spatial_tree.hpp"

#include <Random123/threefry.h>

#include <arbor/arbexcept.hpp>
#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/util/unique_any.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace arb {

namespace {
// We only need minimal hash collisions and good spread over the hash range, because this will be
// used as input for random123, which then provides all desired hash properties.
// std::hash is implementation dependent, so we define our own for reproducibility.

std::uint64_t simple_string_hash(const std::string_view& s) {
    // use fnv1a hash algorithm
    constexpr std::uint64_t prime = 1099511628211ull;
    std::uint64_t h = 14695981039346656037ull;

    for (auto c: s) {
        h ^= c;
        h *= prime;
    }

    return h;
}

struct distributed_site_info {
    cell_gid_type gid = 0;
    cell_lid_type lid = 0;
    cell_kind kind = cell_kind::cable;
    cell_gid_type label_start_idx = 0;
    mlocation location = mlocation();
    mpoint global_location = mpoint();
    network_hash_type hash = 0;
};

struct site_mapping {
    std::vector<distributed_site_info> sites;
    std::vector<char> labels;
    std::unordered_map<std::string, cell_gid_type> label_map;

    site_mapping() = default;

    inline std::size_t size() const { return sites.size(); }

    void insert(const network_full_site_info& s) {
        const auto insert_pair = label_map.insert({std::string(s.label), labels.size()});
        // append label if not contained in labels
        if (insert_pair.second) {
            labels.insert(labels.end(), s.label.begin(), s.label.end());
            labels.push_back('\0');
        }
        sites.emplace_back(distributed_site_info{s.gid,
            s.lid,
            s.kind,
            insert_pair.first->second,
            s.location,
            s.global_location,
            s.hash});
    }

    void insert(const site_mapping& m) {
        for(std::size_t idx = 0; idx < m.size(); ++idx) {
            this->insert(m.get_site(idx));
        }
    }

    network_full_site_info get_site(std::size_t idx) const {
        const auto& s = this->sites.at(idx);

        network_full_site_info info;
        info.gid = s.gid;
        info.lid = s.lid;
        info.kind = s.kind;
        info.label = labels.data() + s.label_start_idx;
        info.location = s.location;
        info.global_location = s.global_location;
        info.hash = s.hash;

        return info;
    }
};

void push_back(const domain_decomposition& dom_dec,
    std::vector<connection>& vec,
    const network_full_site_info& src,
    const network_full_site_info& dest,
    double weight,
    double delay) {
    vec.emplace_back(connection{{src.gid, src.lid}, dest.lid, (float)weight, (float)delay, dom_dec.index_on_domain(dest.gid)});
}

void push_back(const domain_decomposition&,
    std::vector<network_connection_info>& vec,
    const network_full_site_info& src,
    const network_full_site_info& dest,
    double weight,
    double delay) {
    vec.emplace_back(network_connection_info{
        network_site_info{
            src.gid, src.kind, std::string(src.label), src.location, src.global_location},
        network_site_info{
            dest.gid, dest.kind, std::string(dest.label), dest.location, dest.global_location},
        weight,
        delay});
}

template <typename ConnectionType>
std::vector<ConnectionType> generate_network_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec) {
    const auto description_opt = rec.network_description();
    if (!description_opt.has_value()) return {};

    const distributed_context& distributed = *(ctx->distributed);

    const auto& description = description_opt.value();

    const auto selection_ptr = thingify(description.selection, description.dict);
    const auto weight_ptr = thingify(description.weight, description.dict);
    const auto delay_ptr = thingify(description.delay, description.dict);

    const auto& selection = *selection_ptr;
    const auto& weight = *weight_ptr;
    const auto& delay = *delay_ptr;

    std::unordered_map<cell_kind, std::vector<cell_gid_type>> gids_by_kind;

    for (const auto& group: dom_dec.groups()) {
        auto& gids = gids_by_kind[group.kind];
        for (const auto& gid: group.gids) { gids.emplace_back(gid); }
    }

    const auto num_batches = ctx->thread_pool->get_num_threads();
    std::vector<site_mapping> src_site_batches(num_batches);
    std::vector<site_mapping> dest_site_batches(num_batches);

    for (const auto& [kind, gids]: gids_by_kind) {
        const auto batch_size = (gids.size() + num_batches - 1) / num_batches;
        // populate network sites for source and destination
        if (kind == cell_kind::cable) {
            const auto& cable_gids = gids;
            threading::parallel_for::apply(
                0, cable_gids.size(), batch_size, ctx->thread_pool.get(), [&](int i) {
                    const auto batch_idx = ctx->thread_pool->get_current_thread_id().value();
                    auto& src_sites = src_site_batches[batch_idx];
                    auto& dest_sites = dest_site_batches[batch_idx];
                    const auto gid = cable_gids[i];
                    const auto kind = rec.get_cell_kind(gid);
                    // We need access to morphology, so the cell is create directly
                    cable_cell cell;
                    try {
                        cell = util::any_cast<cable_cell&&>(rec.get_cell_description(gid));
                    }
                    catch (std::bad_any_cast&) {
                        throw bad_cell_description(kind, gid);
                    }

                    auto lid_to_label =
                        [](const std::unordered_multimap<cell_tag_type, lid_range>& map,
                            cell_lid_type lid) -> const cell_tag_type& {
                        for (const auto& [label, range]: map) {
                            if (lid >= range.begin && lid < range.end) return label;
                        }
                        throw arbor_internal_error("unkown lid");
                    };

                    place_pwlin location_resolver(cell.morphology(), rec.get_cell_isometry(gid));

                    // check all synapses of cell for potential destination

                    for (const auto& [_, placed_synapses]: cell.synapses()) {
                        for (const auto& p_syn: placed_synapses) {
                            const auto& label = lid_to_label(cell.synapse_ranges(), p_syn.lid);

                            if (selection.select_destination(cell_kind::cable, gid, label)) {
                                const mpoint point = location_resolver.at(p_syn.loc);
                                dest_sites.insert(
                                    {gid, p_syn.lid, cell_kind::cable, label, p_syn.loc, point});
                            }
                        }
                    }

                    // check all detectors of cell for potential source
                    for (const auto& p_det: cell.detectors()) {
                        const auto& label = lid_to_label(cell.detector_ranges(), p_det.lid);
                        if (selection.select_source(cell_kind::cable, gid, label)) {
                            const mpoint point = location_resolver.at(p_det.loc);
                            src_sites.insert(
                                {gid, p_det.lid, cell_kind::cable, label, p_det.loc, point});
                        }
                    }
                });
        }
        else {
            // Assuming all other cell types do not have a morphology. We can use label
            // resolution through factory and set local position to 0.
            auto factory = cell_kind_implementation(kind, backend_kind::multicore, *ctx, 0);

            // We only need the label ranges
            cell_label_range sources, destinations;
            std::ignore = factory(gids, rec, sources, destinations);

            auto& src_sites = src_site_batches[0];
            auto& dest_sites = dest_site_batches[0];

            std::size_t source_label_offset = 0;
            std::size_t destination_label_offset = 0;
            for (std::size_t i = 0; i < gids.size(); ++i) {
                const auto gid = gids[i];
                const auto iso = rec.get_cell_isometry(gid);
                const auto point = iso.apply(mpoint{0.0, 0.0, 0.0, 0.0});
                const auto num_source_labels = sources.sizes().at(i);
                const auto num_destination_labels = destinations.sizes().at(i);

                // Iterate over each source label for current gid
                for (std::size_t j = source_label_offset;
                     j < source_label_offset + num_source_labels;
                     ++j) {
                    const auto& label = sources.labels().at(j);
                    const auto& range = sources.ranges().at(j);
                    for (auto lid = range.begin; lid < range.end; ++lid) {
                        if (selection.select_source(kind, gid, label)) {
                            src_sites.insert({gid, lid, kind, label, {0, 0.0}, point});
                        }
                    }
                }

                // Iterate over each destination label for current gid
                for (std::size_t j = destination_label_offset;
                     j < destination_label_offset + num_destination_labels;
                     ++j) {
                    const auto& label = destinations.labels().at(j);
                    const auto& range = destinations.ranges().at(j);
                    for (auto lid = range.begin; lid < range.end; ++lid) {
                        if (selection.select_destination(kind, gid, label)) {
                            dest_sites.insert({gid, lid, kind, label, {0, 0.0}, point});
                        }
                    }
                }

                source_label_offset += num_source_labels;
                destination_label_offset += num_destination_labels;
            }
        }
    }

    site_mapping& src_sites = src_site_batches.front();

    // combine src batches
    for (std::size_t batch_idx = 1; batch_idx < src_site_batches.size(); ++batch_idx) {

        for (std::size_t i = 0; i < src_site_batches[batch_idx].size(); ++i) {
            src_sites.insert(src_site_batches[batch_idx].get_site(i));
        }
    }

    // create octree
    std::vector<network_full_site_info> network_dest_sites;
    network_dest_sites.reserve(dest_site_batches[0].size() * num_batches);
    for (const auto& dest_sites: dest_site_batches) {
        for (std::size_t i = 0; i < dest_sites.size(); ++i) {
            network_dest_sites.emplace_back(dest_sites.get_site(i));
        }
    }
    const std::size_t max_depth = selection.max_distance().has_value() ? 10 : 1;
    const std::size_t max_leaf_size = 100;
    spatial_tree<network_full_site_info, 3> local_dest_tree(max_depth,
        max_leaf_size,
        std::move(network_dest_sites),
        [](const network_full_site_info& info)
            -> spatial_tree<network_full_site_info, 3>::point_type {
            return {info.global_location.x, info.global_location.y, info.global_location.z};
        });

    // select connections
    std::vector<std::vector<ConnectionType>> connection_batches(num_batches);

    auto sample_sources = [&](const util::range<distributed_site_info*>& source_range,
                              const util::range<char*>& label_range) {
        const auto batch_size = (source_range.size() + num_batches - 1) / num_batches;
        threading::parallel_for::apply(
            0, source_range.size(), batch_size, ctx->thread_pool.get(), [&](int i) {
                const auto& s = source_range[i];
                const auto batch_idx = ctx->thread_pool->get_current_thread_id().value();
                auto& connections = connection_batches[batch_idx];
                network_full_site_info src;
                src.gid = s.gid;
                src.lid = s.lid;
                src.kind = s.kind;
                src.label = label_range.data() + s.label_start_idx;
                src.location = s.location;
                src.global_location = s.global_location;
                src.hash = s.hash;

                auto sample = [&](const network_full_site_info& dest) {
                    if (selection.select_connection(src, dest)) {
                        const auto w = weight.get(src, dest);
                        const auto d = delay.get(src, dest);

                        push_back(dom_dec, connections, src, dest, w, d);
                    }
                };

                if (selection.max_distance().has_value()) {
                    const double d = selection.max_distance().value();
                    local_dest_tree.bounding_box_for_each(
                        decltype(local_dest_tree)::point_type{src.global_location.x - d,
                            src.global_location.y - d,
                            src.global_location.z - d},
                        decltype(local_dest_tree)::point_type{src.global_location.x + d,
                            src.global_location.y + d,
                            src.global_location.z + d},
                        sample);
                }
                else { local_dest_tree.for_each(sample); }
            });
    };

    distributed_for_each(sample_sources,
        distributed,
        util::range_view(src_sites.sites),
        util::range_view(src_sites.labels));

    // concatenate
    auto connections = std::move(connection_batches.front());
    for (std::size_t i = 1; i < connection_batches.size(); ++i) {
        connections.insert(
            connections.end(), connection_batches[i].begin(), connection_batches[i].end());
    }
    return connections;
}

}  // namespace

network_full_site_info::network_full_site_info(cell_gid_type gid,
    cell_lid_type lid,
    cell_kind kind,
    std::string_view label,
    mlocation location,
    mpoint global_location):
    gid(gid),
    lid(lid),
    kind(kind),
    label(std::move(label)),
    location(location),
    global_location(global_location) {

    std::uint64_t label_hash = simple_string_hash(this->label);
    static_assert(sizeof(decltype(mlocation::pos)) == sizeof(std::uint64_t));
    std::uint64_t loc_pos_hash = *reinterpret_cast<const std::uint64_t*>(&location.pos);

    // Initial seed. Changes will affect reproducibility of generated network connections.
    constexpr std::uint64_t seed = 984293;

    using rand_type = r123::Threefry4x64;
    const rand_type::ctr_type seed_input = {{seed, 2 * seed, 3 * seed, 4 * seed}};
    const rand_type::key_type key = {{gid, label_hash, location.branch, loc_pos_hash}};

    rand_type gen;
    hash = gen(seed_input, key)[0];
}

std::vector<connection> generate_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec) {
    return generate_network_connections<connection>(rec, ctx, dom_dec);
}

ARB_ARBOR_API std::vector<network_connection_info> generate_network_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec) {
    auto connections = generate_network_connections<network_connection_info>(rec, ctx, dom_dec);

    // generated connections may have different order each time due to multi-threading.
    // Sort before returning to user for reproducibility.
    std::sort(connections.begin(), connections.end());

    return connections;
}

ARB_ARBOR_API std::vector<network_connection_info> generate_network_connections(const recipe& rec) {
    auto ctx = arb::make_context();
    auto decomp = arb::partition_load_balance(rec, ctx);

    return generate_network_connections(rec, ctx, decomp);
}

}  // namespace arb
