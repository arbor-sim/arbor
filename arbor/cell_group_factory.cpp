#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

#include "benchmark_cell_group.hpp"
#include "cell_group.hpp"
#include "fvm_lowered_cell.hpp"
#include "lif_cell_group.hpp"
#include "mc_cell_group.hpp"
#include "spike_source_cell_group.hpp"

namespace arb {

cell_group_ptr cell_group_factory(const recipe& rec, const group_description& group) {
    switch (group.kind) {
    case cell_kind::cable1d_neuron:
        return make_cell_group<mc_cell_group>(group.gids, rec, make_fvm_lowered_cell(group.backend));

    case cell_kind::spike_source:
        return make_cell_group<spike_source_cell_group>(group.gids, rec);

    case cell_kind::lif_neuron:
        return make_cell_group<lif_cell_group>(group.gids, rec);

    case cell_kind::benchmark:
        return make_cell_group<benchmark_cell_group>(group.gids, rec);

    default:
        throw arbor_internal_error("cell_group_factory: unknown cell kind");
    }
}

} // namespace arb
