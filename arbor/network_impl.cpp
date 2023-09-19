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
struct network_site_info_extended {
    network_site_info_extended(network_site_info info, cell_lid_type lid):
        info(std::move(info)),
        lid(lid) {}

    network_site_info info;
    cell_lid_type lid;
};

void push_back(const domain_decomposition& dom_dec,
    std::vector<connection>& vec,
    const network_site_info_extended& source,
    const network_site_info_extended& target,
    double weight,
    double delay) {
    vec.emplace_back(connection{{source.info.gid, source.lid},
        target.lid,
        (float)weight,
        (float)delay,
        dom_dec.index_on_domain(target.info.gid)});
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
    std::vector<std::vector<network_site_info_extended>> src_site_batches(num_batches);
    std::vector<std::vector<network_site_info_extended>> tgt_site_batches(num_batches);

    for (const auto& [kind, gids]: gids_by_kind) {
        const auto batch_size = (gids.size() + num_batches - 1) / num_batches;
        // populate network sites for source and target
        if (kind == cell_kind::cable) {
            const auto& cable_gids = gids;
            threading::parallel_for::apply(
                0, cable_gids.size(), batch_size, ctx->thread_pool.get(), [&](int i) {
                    const auto batch_idx = ctx->thread_pool->get_current_thread_id().value();
                    auto& src_sites = src_site_batches[batch_idx];
                    auto& tgt_sites = tgt_site_batches[batch_idx];
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
                        [](const std::unordered_multimap<hash_type, lid_range>& map,
                            cell_lid_type lid) -> hash_type {
                        for (const auto& [label, range]: map) {
                            if (lid >= range.begin && lid < range.end) return label;
                        }
                        throw arbor_internal_error("unkown lid");
                    };

                    place_pwlin location_resolver(cell.morphology(), rec.get_cell_isometry(gid));

                    // check all synapses of cell for potential target

                    for (const auto& [_, placed_synapses]: cell.synapses()) {
                        for (const auto& p_syn: placed_synapses) {
                            const auto& label = lid_to_label(cell.synapse_ranges(), p_syn.lid);

                            if (selection.select_target(cell_kind::cable, gid, label)) {
                                const mpoint point = location_resolver.at(p_syn.loc);
                                tgt_sites.emplace_back(
                                    network_site_info{
                                        gid, cell_kind::cable, label, p_syn.loc, point},
                                    p_syn.lid);
                            }
                        }
                    }

                    // check all detectors of cell for potential source
                    for (const auto& p_det: cell.detectors()) {
                        const auto& label = lid_to_label(cell.detector_ranges(), p_det.lid);
                        if (selection.select_source(cell_kind::cable, gid, label)) {
                            const mpoint point = location_resolver.at(p_det.loc);
                            src_sites.emplace_back(
                                network_site_info{gid, cell_kind::cable, label, p_det.loc, point},
                                p_det.lid);
                        }
                    }
                });
        }
        else {
            // Assuming all other cell types do not have a morphology. We can use label
            // resolution through factory and set local position to 0.
            auto factory = cell_kind_implementation(kind, backend_kind::multicore, *ctx, 0);

            // We only need the label ranges
            cell_label_range sources, targets;
            std::ignore = factory(gids, rec, sources, targets);

            auto& src_sites = src_site_batches[0];
            auto& tgt_sites = tgt_site_batches[0];

            std::size_t source_label_offset = 0;
            std::size_t target_label_offset = 0;
            for (std::size_t i = 0; i < gids.size(); ++i) {
                const auto gid = gids[i];
                const auto iso = rec.get_cell_isometry(gid);
                const auto point = iso.apply(mpoint{0.0, 0.0, 0.0, 0.0});
                const auto num_source_labels = sources.sizes.at(i);
                const auto num_target_labels = targets.sizes.at(i);

                // Iterate over each source label for current gid
                for (std::size_t j = source_label_offset;
                     j < source_label_offset + num_source_labels;
                     ++j) {
                    const auto& label = sources.labels.at(j);
                    const auto& range = sources.ranges.at(j);
                    for (auto lid = range.begin; lid < range.end; ++lid) {
                        if (selection.select_source(kind, gid, label)) {
                            src_sites.emplace_back(
                                network_site_info{gid, kind, label, mlocation{0, 0.0}, point}, lid);
                        }
                    }
                }

                // Iterate over each target label for current gid
                for (std::size_t j = target_label_offset;
                     j < target_label_offset + num_target_labels;
                     ++j) {
                    const auto& label = targets.labels.at(j);
                    const auto& range = targets.ranges.at(j);
                    for (auto lid = range.begin; lid < range.end; ++lid) {
                        if (selection.select_target(kind, gid, label)) {
                            tgt_sites.emplace_back(
                                network_site_info{gid, kind, label, mlocation{0, 0.0}, point}, lid);
                        }
                    }
                }

                source_label_offset += num_source_labels;
                target_label_offset += num_target_labels;
            }
        }
    }


    auto src_sites = std::move(src_site_batches.back());
    src_site_batches.pop_back();
    for (const auto& batch: src_site_batches)
            src_sites.insert(src_sites.end(), batch.begin(), batch.end());

    auto tgt_sites = std::move(tgt_site_batches.back());
    tgt_site_batches.pop_back();
    for (const auto& batch: tgt_site_batches)
            tgt_sites.insert(tgt_sites.end(), batch.begin(), batch.end());

    // create octree
    const std::size_t max_depth = selection.max_distance().has_value() ? 10 : 1;
    const std::size_t max_leaf_size = 100;
    spatial_tree<network_site_info_extended, 3> local_tgt_tree(max_depth,
        max_leaf_size,
        std::move(tgt_sites),
        [](const network_site_info_extended& ex)
            -> spatial_tree<network_site_info_extended, 3>::point_type {
            return {
                ex.info.global_location.x, ex.info.global_location.y, ex.info.global_location.z};
        });

    // select connections
    std::vector<std::vector<ConnectionType>> connection_batches(num_batches);

    auto sample_sources = [&](const util::range<network_site_info_extended*>& source_range) {
        const auto batch_size = (source_range.size() + num_batches - 1) / num_batches;
        threading::parallel_for::apply(
            0, source_range.size(), batch_size, ctx->thread_pool.get(), [&](int i) {
                const auto& source = source_range[i];
                const auto batch_idx = ctx->thread_pool->get_current_thread_id().value();
                auto& connections = connection_batches[batch_idx];

                auto sample = [&](const network_site_info_extended& target) {
                    if (selection.select_connection(source.info, target.info)) {
                        const auto w = weight.get(source.info, target.info);
                        const auto d = delay.get(source.info, target.info);

                        push_back(dom_dec, connections, source, target, w, d);
                    }
                };

                if (selection.max_distance().has_value()) {
                    const double d = selection.max_distance().value();
                    local_tgt_tree.bounding_box_for_each(
                        decltype(local_tgt_tree)::point_type{source.info.global_location.x - d,
                            source.info.global_location.y - d,
                            source.info.global_location.z - d},
                        decltype(local_tgt_tree)::point_type{source.info.global_location.x + d,
                            source.info.global_location.y + d,
                            source.info.global_location.z + d},
                        sample);
                }
                else { local_tgt_tree.for_each(sample); }
            });
    };

    distributed_for_each(sample_sources, distributed, util::range_view(src_sites));

    // concatenate
    auto connections = std::move(connection_batches.front());
    for (std::size_t i = 1; i < connection_batches.size(); ++i) {
        connections.insert(
            connections.end(), connection_batches[i].begin(), connection_batches[i].end());
    }
    return connections;
}

}  // namespace

std::vector<connection> generate_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec) {
    return generate_network_connections<connection>(rec, ctx, dom_dec);
}

// ARB_ARBOR_API std::vector<network_connection_info> generate_network_connections(const recipe& rec,
//     const context& ctx,
//     const domain_decomposition& dom_dec) {
//     auto connections = generate_network_connections<network_connection_info>(rec, ctx, dom_dec);

//     // generated connections may have different order each time due to multi-threading.
//     // Sort before returning to user for reproducibility.
//     std::sort(connections.begin(), connections.end());

//     return connections;
// }

// ARB_ARBOR_API std::vector<network_connection_info> generate_network_connections(const recipe& rec) {
//     auto ctx = arb::make_context();
//     auto decomp = arb::partition_load_balance(rec, ctx);

//     return generate_network_connections(rec, ctx, decomp);
// }

}  // namespace arb
