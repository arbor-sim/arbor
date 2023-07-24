#include "network_impl.hpp"
#include "cell_group_factory.hpp"
#include "communication/distributed_for_each.hpp"
#include "label_resolution.hpp"
#include "network_impl.hpp"
#include "threading/threading.hpp"
#include "util/range.hpp"
#include "util/spatial_tree.hpp"

#include <Random123/threefry.h>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/lif_cell.hpp>
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

void push_back(std::vector<connection>& vec,
    const network_full_site_info& src,
    const network_full_site_info& dest,
    double weight,
    double delay) {
    vec.emplace_back(connection({src.gid, src.lid}, {dest.gid, dest.lid}, weight, delay));
}

void push_back(std::vector<network_connection_info>& vec,
    const network_full_site_info& src,
    const network_full_site_info& dest,
    double weight,
    double delay) {
    vec.emplace_back(network_connection_info{
        network_site_info{
            src.gid, src.kind, std::string(src.label), src.location, src.global_location},
        network_site_info{
            dest.gid, dest.kind, std::string(dest.label), dest.location, dest.global_location}});
}

template <typename ConnectionType>
std::vector<ConnectionType> generate_network_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec) {
    const auto description_opt = rec.network_description();
    if (!description_opt.has_value()) return {};

    const distributed_context& distributed = *(ctx->distributed);

    const auto& description = description_opt.value();

    site_mapping src_sites, dest_sites;
    std::mutex src_sites_mutex, dest_sites_mutex;

    const auto selection_ptr = thingify(description.selection, description.dict);
    const auto weight_ptr = thingify(description.weight, description.dict);
    const auto delay_ptr = thingify(description.delay, description.dict);

    const auto& selection = *selection_ptr;
    const auto& weight = *weight_ptr;
    const auto& delay = *delay_ptr;

    // populate network sites for source and destination
    for (const auto& group: dom_dec.groups()) {
        switch (group.kind) {
        case cell_kind::cable: {
            // We need access to morphology, so the cell is create directly
            threading::parallel_for::apply(
                0, group.gids.size(), ctx->thread_pool.get(), [&](int i) {
                    const auto gid = group.gids[i];
                    cable_cell cell;
                    try {
                        cell = util::any_cast<cable_cell&&>(rec.get_cell_description(gid));
                    }
                    catch (std::bad_any_cast&) {
                        throw bad_cell_description(rec.get_cell_kind(gid), gid);
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
                                std::lock_guard<std::mutex> guard(dest_sites_mutex);
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
                            std::lock_guard<std::mutex> guard(src_sites_mutex);
                            src_sites.insert(
                                {gid, p_det.lid, cell_kind::cable, label, p_det.loc, point});
                        }
                    }
                });
        } break;
        default: {
            // Assuming all other cell types do not have a morphology. We can use label resolution
            // through factory and set local position to 0.
            auto factory = cell_kind_implementation(group.kind, group.backend, *ctx, 0);

            // We only need the label ranges
            cell_label_range sources, destinations;
            std::ignore = factory(group.gids, rec, sources, destinations);

            std::size_t source_label_offset = 0;
            std::size_t destination_label_offset = 0;
            for (std::size_t i = 0; i < group.gids.size(); ++i) {
                const auto gid = group.gids[i];
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
                        if (selection.select_source(group.kind, gid, label)) {
                            std::lock_guard<std::mutex> guard(src_sites_mutex);
                            src_sites.insert({gid, lid, group.kind, label, {0, 0.0}, point});
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
                        if (selection.select_destination(group.kind, gid, label)) {
                            std::lock_guard<std::mutex> guard(dest_sites_mutex);
                            dest_sites.insert({gid, lid, group.kind, label, {0, 0.0}, point});
                        }
                    }
                }

                source_label_offset += num_source_labels;
                destination_label_offset += num_destination_labels;
            }

        } break;
        }
    }

    // create octree
    std::vector<network_full_site_info> network_dest_sites;
    network_dest_sites.reserve(dest_sites.size());
    for (std::size_t i = 0; i < dest_sites.size(); ++i) {
        network_dest_sites.emplace_back(dest_sites.get_site(i));
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
    std::vector<ConnectionType> connections;
    std::mutex connections_mutex;

    auto sample_sources = [&](const util::range<distributed_site_info*>& source_range,
                              const util::range<char*>& label_range) {
        threading::parallel_for::apply(0, source_range.size(), ctx->thread_pool.get(), [&](int i) {
            const auto& s = source_range[i];
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

                    std::lock_guard<std::mutex> guard(connections_mutex);
                    push_back(connections, src, dest, w, d);
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
        util::make_range(src_sites.sites.begin(), src_sites.sites.end()),
        util::make_range(src_sites.labels.begin(), src_sites.labels.end()));

    // distributed_for_each(sample_sources, distributed, src_sites.sites, src_sites.labels);

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

    // generated connections may have different orer each time due to multi-threading.
    // sort before returning to user for reproducibility.
    std::sort(connections.begin(), connections.end());

    return connections;
}

}  // namespace arb
