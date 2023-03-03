#include "network_generation.hpp"
#include "util/spatial_tree.hpp"

#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/util/unique_any.hpp>
#include <cstddef>

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
