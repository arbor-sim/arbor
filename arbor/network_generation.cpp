#include "network_generation.hpp"
#include "cell_group_factory.hpp"
#include "communication/distributed_for_each.hpp"
#include "network_impl.hpp"
#include "util/range.hpp"
#include "util/spatial_tree.hpp"

#include <Random123/threefry.h>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/util/unique_any.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
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
    mpoint global_location = mpoint();
    network_hash_type hash = 0;
};


struct site_mapping {
    std::vector<distributed_site_info> sites;
    std::vector<char> labels;
    std::unordered_map<std::string, cell_gid_type> label_map;

    site_mapping() = default;

    inline std::size_t size() const { return sites.size(); }

    void insert(const network_site_info& s) {
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

    network_site_info get_site(std::size_t idx) const {
        const auto& s = this->sites.at(idx);

        network_site_info info;
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

}  // namespace

std::vector<connection> generate_network_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec) {
    const auto description_opt = rec.network_description();
    if (!description_opt.has_value()) return {};

    const distributed_context& distributed = *(ctx->distributed);

    const auto& description = description_opt.value();

    site_mapping src_sites, dest_sites;

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

                place_pwlin location_resolver(cell.morphology(), rec.get_cell_isometry(gid));

                // check all synapses of cell for potential destination

                for (const auto& [_, placed_synapses]: cell.synapses()) {
                    for (const auto& p_syn: placed_synapses) {
                        // TODO check if tag correct
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
                    // TODO check if tag correct
                    const auto& label = lid_to_label(cell.detector_ranges(), p_det.lid);
                    if (selection.select_source(cell_kind::cable, gid, label)) {
                        const mpoint point = location_resolver.at(p_det.loc);
                        src_sites.insert(
                            {gid, p_det.lid, cell_kind::cable, label, p_det.loc, point});
                    }
                }
            }
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
    std::vector<network_site_info> network_dest_sites;
    network_dest_sites.reserve(dest_sites.size());
    for(std::size_t i = 0; i < dest_sites.size(); ++i) {
        network_dest_sites.emplace_back(dest_sites.get_site(i));
    }
    const std::size_t max_depth = selection.max_distance().has_value() ? 10 : 1;
    const std::size_t max_leaf_size = 100;
    spatial_tree<network_site_info, 3> local_dest_tree(max_depth,
        max_leaf_size,
        std::move(network_dest_sites),
        [](const network_site_info& info) -> spatial_tree<network_site_info, 3>::point_type {
            return {info.global_location.x, info.global_location.y, info.global_location.z};
        });

    // select connections
    std::vector<connection> connections;

    auto sample_sources = [&](const util::range<distributed_site_info*>& source_range,
                              const util::range<char*>& label_range) {
        for (const auto& s: source_range) {
            network_site_info src;
            src.gid = s.gid;
            src.lid = s.lid;
            src.kind = s.kind;
            src.label = label_range.data() + s.label_start_idx;
            src.location = s.location;
            src.global_location = s.global_location;
            src.hash = s.hash;

            auto sample = [&](const network_site_info& dest) {
                if (selection.select_connection(src, dest)) {
                    connections.emplace_back(connection({src.gid, src.lid},
                        {dest.gid, dest.lid},
                        weight.get(src, dest),
                        delay.get(src, dest)));
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
        }
    };

    distributed_for_each(sample_sources, distributed, src_sites.sites, src_sites.labels);

    return connections;
}

}  // namespace arb
