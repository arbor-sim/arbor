#include <vector>

#include <backends.hpp>
#include <benchmark_cell_group.hpp>
#include <cell_group.hpp>
#include <domain_decomposition.hpp>
#include <fvm_lowered_cell.hpp>
#include <lif_cell_group.hpp>
#include <mc_cell_group.hpp>
#include <recipe.hpp>
#include <spike_source_cell_group.hpp>
#include <util/unique_any.hpp>

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
        throw std::runtime_error("unknown cell kind");
    }
}

} // namespace arb
