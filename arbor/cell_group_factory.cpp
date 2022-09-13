#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/recipe.hpp>

#include "benchmark_cell_group.hpp"
#include "cell_group.hpp"
#include "cell_group_factory.hpp"
#include "execution_context.hpp"
#include "fvm_lowered_cell.hpp"
#include "lif_cell_group.hpp"
#include "mc_cell_group.hpp"
#include "spike_source_cell_group.hpp"

namespace arb {

template <typename Impl, typename... Args>
cell_group_ptr make_cell_group(Args&&... args) {
    return cell_group_ptr(new Impl(std::forward<Args>(args)...));
}

ARB_ARBOR_API cell_group_factory cell_kind_implementation(
        cell_kind ck, backend_kind bk, const execution_context& ctx, arb_seed_type seed)
{
    using gid_vector = std::vector<cell_gid_type>;

    switch (ck) {
    case cell_kind::cable:
        return [bk, ctx, seed](const gid_vector& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets) {
            return make_cell_group<mc_cell_group>(gids, rec, cg_sources, cg_targets, make_fvm_lowered_cell(bk, ctx, seed));
        };

    case cell_kind::spike_source:
        if (bk!=backend_kind::multicore) break;

        return [](const gid_vector& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets) {
            return make_cell_group<spike_source_cell_group>(gids, rec, cg_sources, cg_targets);
        };

    case cell_kind::lif:
        if (bk!=backend_kind::multicore) break;

        return [](const gid_vector& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets) {
            return make_cell_group<lif_cell_group>(gids, rec, cg_sources, cg_targets);
        };

    case cell_kind::benchmark:
        if (bk!=backend_kind::multicore) break;

        return [](const gid_vector& gids, const recipe& rec, cell_label_range& cg_sources, cell_label_range& cg_targets) {
            return make_cell_group<benchmark_cell_group>(gids, rec, cg_sources, cg_targets);
        };

    default: ;
    }

    return cell_group_factory{}; // empty function => not supported
}

} // namespace arb
