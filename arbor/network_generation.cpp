#include "network_generation.hpp"
#include "network_impl.hpp"
#include "util/spatial_tree.hpp"

#include <Random123/threefry.h>

#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/util/unique_any.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace arb {

namespace {
struct distributed_site_info {
    cell_gid_type gid = 0;
    cell_lid_type lid = 0;
    cell_kind kind = cell_kind::cable;
    cell_gid_type label_start_idx = 0;
    mlocation location = mlocation();
    network_location global_location = network_location();
    network_hash_type hash = 0;
};

struct distributed_site_mapping {
    std::vector<distributed_site_info> sites;
    std::string labels;

    distributed_site_mapping() = default;

    explicit distributed_site_mapping(const std::vector<network_site_info>& net_sites) {
        std::unordered_map<std::string_view, cell_gid_type> label_to_start_idx;

        for (const auto& s: net_sites) {
            const auto insert_pair = label_to_start_idx.insert({s.label, labels.size()});
            // append label if not contained in labels
            if (insert_pair.second) {
                labels.append(s.label);
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
    }

    std::string_view label_at_site(const distributed_site_info& si) {
        return labels.c_str() + si.label_start_idx;
    }
};

template <typename FUNC>
void distributed_for_each_site(const distributed_context& distributed,
    const std::vector<network_site_info>& src_sites,
    FUNC f) {
    if(distributed.size() > 1) {
        distributed_site_mapping mapping(src_sites);

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

        distributed_site_mapping recv_mapping;
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
                f(network_site_info{
                    s.gid, s.lid, s.kind, mapping.label_at_site(s), s.location, s.global_location});
            }

            request_sites.finalize();
            request_labels.finalize();

            std::swap(mapping, recv_mapping);

            current_idx = next_idx;
        }

        for (std::size_t site_idx = 0; site_idx < num_sites_per_rank[current_idx]; ++site_idx) {
            const auto& s = mapping.sites[site_idx];
            f(network_site_info{
                s.gid, s.lid, s.kind, mapping.label_at_site(s), s.location, s.global_location});
        }
    }
    else {
        for (const auto& s: src_sites) { f(s); }
    }
}

}  // namespace

std::vector<connection> generate_network_connections(const recipe& rec,
    const distributed_context& distributed,
    const domain_decomposition& dom_dec) {
    const auto description_opt = rec.network_description();
    if(!description_opt.has_value()) return {};

    const auto& description = description_opt.value();

    std::vector<network_site_info> src_sites, dest_sites;

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
            cable_cell cell;
            for (const auto& gid: group.gids) {
                try {
                    cell = util::any_cast<cable_cell&&>(rec.get_cell_description(gid));
                }
                catch (std::bad_any_cast&) {
                    throw bad_cell_description(rec.get_cell_kind(gid), gid);
                }

                auto lid_to_label = [](const std::unordered_multimap<cell_tag_type, lid_range>& map,
                                        cell_lid_type lid) -> const cell_tag_type& {
                    for (const auto& [label, range]: map) {
                        if (lid >= range.begin && lid < range.end) return label;
                    }
                    throw arbor_internal_error("unkown lid");
                };

                place_pwlin location_resolver(cell.morphology());

                // check all synapses of cell for potential destination

                for (const auto& [_, placed_synapses]: cell.synapses()) {
                    for (const auto& p_syn: placed_synapses) {
                        // TODO check if tag correct
                        const auto& label = lid_to_label(cell.synapse_ranges(), p_syn.lid);

                        if (selection.select_destination(cell_kind::cable, gid, label)) {
                            // TODO: compute rotation and global offset
                            const mpoint point = location_resolver.at(p_syn.loc);
                            network_location global_location = {point.x, point.y, point.z};
                            dest_sites.emplace_back(gid,
                                p_syn.lid,
                                cell_kind::cable,
                                label,
                                p_syn.loc,
                                global_location);
                        }
                    }
                }

                // check all detectors of cell for potential source
                for (const auto& p_det: cell.detectors()) {
                    // TODO check if tag correct
                    const auto& label = lid_to_label(cell.synapse_ranges(), p_det.lid);
                    if (selection.select_destination(cell_kind::cable, gid, label)) {
                        // TODO: compute rotation and global offset
                        const mpoint point = location_resolver.at(p_det.loc);
                        network_location global_location = {point.x, point.y, point.z};
                        src_sites.emplace_back(
                            gid, p_det.lid, cell_kind::cable, label, p_det.loc, global_location);
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

    // create octree
    const std::size_t max_depth = selection.max_distance().has_value() ? 10 : 1;
    const std::size_t max_leaf_size = 100;
    spatial_tree<network_site_info, 3> local_dest_tree(
        max_depth, max_leaf_size, std::move(dest_sites), [](const network_site_info& info) {
            return info.global_location;
        });

    // select connections
    std::vector<connection> connections;

    auto sample_destinations = [&] (const network_site_info& src) {
        auto sample =  [&] (const network_site_info& dest) {
            if(selection.select_connection(src, dest)) {
                connections.emplace_back(connection({src.gid, src.lid},
                    {dest.gid, dest.lid},
                    weight.get(src, dest),
                    delay.get(src, dest)));
            }
        };

        if(selection.max_distance().has_value()) {
            const double d = selection.max_distance().value();
            local_dest_tree.bounding_box_for_each(network_location{src.global_location[0] - d,
                                                      src.global_location[1] - d,
                                                      src.global_location[2] - d},
                network_location{src.global_location[0] + d,
                    src.global_location[1] + d,
                    src.global_location[2] + d},
                sample);
        } else {
            local_dest_tree.for_each(sample);
        }
    };

    distributed_for_each_site(distributed, src_sites, sample_destinations);

    return connections;
}

}  // namespace arb
