#include "network_generation.hpp"
#include "util/spatial_tree.hpp"

#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/util/unique_any.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

namespace arb {

namespace {
struct dest_site_info {
    cell_gid_type gid;
    cell_lid_type lid;
    network_hash_type hash;
};

struct src_site_info {
    cell_gid_type gid;
    cell_lid_type lid;
    double x, y, z;
    network_hash_type hash;
};


struct site_info {
    cell_gid_type gid;
    cell_gid_type label_id;
    cell_lid_type lid;
    network_location location;
};

struct site_collection {
    std::unordered_map<cell_tag_type, cell_gid_type> label_id_mapping;
    std::vector<site_info> sites;

    inline void add_site(cell_gid_type gid,
        const cell_tag_type& label,
        cell_lid_type lid,
        network_location location) {

        auto insert_it = label_id_mapping.insert({label, label_id_mapping.size()});

        sites.emplace_back(site_info{gid, insert_it.first->second, lid, location});
    }
};

struct site_mapping {
    std::vector<site_info> sites;
    std::string labels;

    site_mapping() = default;

    site_mapping(site_collection collection) {

        std::size_t totalLabelLength = 0;
        for (const auto& [label, _]: collection.label_id_mapping) {
            totalLabelLength += label.size();
        }

        labels.reserve(totalLabelLength + collection.label_id_mapping.size());
        std::vector<cell_gid_type> label_id_to_start_idx(collection.label_id_mapping.size());
        for (const auto& [label, id]: collection.label_id_mapping) {
            label_id_to_start_idx[id] = labels.size();
            labels.append(label);
            labels.push_back('\0');
        }

        for(auto& si : collection.sites) {
            si.label_id = label_id_to_start_idx.at(si.label_id);
        }

        sites = std::move(collection.sites);
    }

    std::string_view label_at_site(const site_info& si) {
        return labels.c_str() + si.label_id;
    }
};

template <typename FUNC>
void distributed_for_each_site(const distributed_context& distributed,
    site_mapping mapping,
    FUNC f) {
    if(distributed.size() > 1) {
        const auto my_rank = distributed.id();
        const auto left_rank = my_rank == 0 ? distributed.size() - 1 : my_rank - 1;
        const auto right_rank = my_rank == distributed.size() - 1 ? 0 : my_rank + 1;

        const auto num_sites_per_rank = distributed.gather_all(mapping.sites.size());
        const auto label_string_size_per_rank = distributed.gather_all(mapping.labels.size());

        const auto max_num_sites =
            *std::max_element(num_sites_per_rank.begin(), num_sites_per_rank.end());
        const auto max_string_size =
            *std::max_element(label_string_size_per_rank.begin(), label_string_size_per_rank.end());

        mapping.sites.resize(max_num_sites);
        mapping.labels.resize(max_string_size);

        site_mapping recv_mapping;
        recv_mapping.sites.resize(max_num_sites);
        recv_mapping.labels.resize(max_string_size);

        auto current_idx = my_rank;

        for(std::size_t step = 0; step < distributed.size() - 1; ++step) {
            const auto next_idx = (current_idx + 1) % distributed.size();
            auto request_sites = distributed.send_recv_nonblocking(num_sites_per_rank[next_idx],
                recv_mapping.sites.data(),
                right_rank,
                num_sites_per_rank[current_idx],
                mapping.sites.data(),
                left_rank,
                0);

            auto request_labels =
                distributed.send_recv_nonblocking(label_string_size_per_rank[next_idx],
                    recv_mapping.labels.data(),
                    right_rank,
                    label_string_size_per_rank[current_idx],
                    mapping.labels.data(),
                    left_rank,
                    1);

            for (std::size_t site_idx = 0; site_idx < num_sites_per_rank[current_idx]; ++site_idx) {
                const auto& s = mapping.sites[site_idx];
                f(s, mapping.label_at_site(s));
            }

            request_sites.finalize();
            request_labels.finalize();

            std::swap(mapping, recv_mapping);

            current_idx = next_idx;
        }

        for (std::size_t site_idx = 0; site_idx < num_sites_per_rank[current_idx]; ++site_idx) {
            const auto& s = mapping.sites[site_idx];
            f(s, mapping.label_at_site(s));
        }
    } else {
        for (const auto& s: mapping.sites) { f(s, mapping.label_at_site(s)); }
    }
}

}  // namespace

std::vector<connection> generate_network_connections(
    const std::vector<network_description>& descriptions,
    const recipe& rec,
    const distributed_context& distributed,
    const domain_decomposition& dom_dec,
    const label_resolution_map& source_resolution_map,
    const label_resolution_map& target_resolution_map) {
    if (descriptions.empty()) return {};

    std::vector<std::vector<src_site_info>> local_src_sites(descriptions.size());
    std::vector<std::vector<std::pair<network_location, dest_site_info>>> local_dest_sites(
        descriptions.size());

    // populate network sites for source and destination
    for (const auto& group: dom_dec.groups()) {
        switch (group.kind) {
        case cell_kind::cable: {
            cable_cell cell;
            for (const auto& gid: group.gids) {
                try {
                    cell = util::any_cast<cable_cell&&>(rec.get_cell_description(gid));
                }
                catch (std::bad_any_cast&) {
                    throw bad_cell_description(rec.get_cell_kind(gid), gid);
                }

                place_pwlin location_resolver(cell.morphology());

                // check all synapses of cell for potential destination
                for (const auto& [name, placed_synapses]: cell.synapses()) {
                    for (const auto& p_syn: placed_synapses) {
                        // TODO: compute rotation and global offset
                        const mpoint point = location_resolver.at(p_syn.loc);
                        network_location location = {point.x, point.y, point.z};
                        // TODO check if tag correct
                        const auto& tag = target_resolution_map.tag_at(gid, p_syn.lid);

                        for (std::size_t i = 0; i < descriptions.size(); ++i) {
                            const auto& desc = descriptions[i];
                            if (desc.dest_selection(
                                    gid, cell_kind::cable, tag, p_syn.loc, location)) {
                                // TODO : compute hash
                                network_hash_type hash = 0;
                                local_dest_sites[i].push_back({location, {gid, p_syn.lid, hash}});
                            }
                        }
                    }
                }

                // check all detectors of cell for potential source
                for (const auto& p_det: cell.detectors()) {
                    // TODO: compute rotation and global offset
                    const mpoint point = location_resolver.at(p_det.loc);
                    network_location location = {point.x, point.y, point.z};
                    // TODO check if tag correct
                    const auto& tag = target_resolution_map.tag_at(gid, p_det.lid);

                    for (std::size_t i = 0; i < descriptions.size(); ++i) {
                        const auto& desc = descriptions[i];
                        if (desc.src_selection(gid, cell_kind::cable, tag, p_det.loc, location)) {
                            // TODO : compute hash
                            network_hash_type hash = 0;
                            local_src_sites[i].push_back(
                                {gid, p_det.lid, location[0], location[1], location[2], hash});
                        }
                    }
                }
            }
        } break;
        case cell_kind::lif: {
            // TODO
            for (const auto& gid: group.gids) {}
        } break;
        case cell_kind::benchmark: {
            // TODO
            for (const auto& gid: group.gids) {}
        } break;
        case cell_kind::spike_source: {
            // TODO
            for (const auto& gid: group.gids) {}
        } break;
        }
    }

    // create octrees
    std::vector<spatial_tree<dest_site_info, 3>> local_dest_trees;
    local_dest_trees.reserve(descriptions.size());
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
        const auto& desc = descriptions[i];
        const std::size_t max_depth = desc.connection_selection.max_distance().has_value() ? 10 : 1;
        local_dest_trees.emplace_back(max_depth, 100, std::move(local_dest_sites[i]));
    }

    // select connections
    std::vector<connection> connections;

    for (std::size_t i = 0; i < descriptions.size(); ++i) {
        const auto& desc = descriptions[i];
        const auto& src_sites = local_src_sites[i];
        const auto& dest_tree = local_dest_trees[i];

        for (const auto& src: src_sites) {
            auto sample_dest = [&](const network_location& dest_loc, const dest_site_info& dest) {
                // TODO precompute distance
                if (desc.connection_selection(
                        src.gid, {src.x, src.y, src.z}, src.hash, dest.gid, dest_loc, dest.hash)) {
                    const double w = desc.weight(
                        src.gid, {src.x, src.y, src.z}, src.hash, dest.gid, dest_loc, dest.hash);
                    const double d = desc.delay(
                        src.gid, {src.x, src.y, src.z}, src.hash, dest.gid, dest_loc, dest.hash);

                    connections.emplace_back(cell_member_type{src.gid, src.lid},
                        cell_member_type{dest.gid, dest.lid},
                        w,
                        d);
                }
            };

            if(desc.connection_selection.max_distance().has_value()) {
                const double d = desc.connection_selection.max_distance().value();
                dest_tree.bounding_box_for_each(network_location{src.x - d, src.y - d, src.z - d},
                    network_location{src.x + d, src.y + d, src.z + d},
                    sample_dest);
            }
            else { dest_tree.for_each(sample_dest); }
        }
    }

    return connections;
}

}  // namespace arb
