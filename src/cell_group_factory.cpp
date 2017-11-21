#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <domain_decomposition.hpp>
#include <dss_cell_group.hpp>
#include <fvm_multicell.hpp>
#include <lif_cell_group_mc.hpp>
#ifdef ARB_WITH_CUDA
#include <lif_cell_group_gpu.hpp>
#endif
#include <mc_cell_group.hpp>
#include <recipe.hpp>
#include <rss_cell_group.hpp>
#include <util/unique_any.hpp>

namespace arb {

using gpu_fvm_cell = mc_cell_group<fvm::fvm_multicell<gpu::backend>>;
using mc_fvm_cell = mc_cell_group<fvm::fvm_multicell<multicore::backend>>;

cell_group_ptr cell_group_factory(const recipe& rec, const group_description& group) {
    switch (group.kind) {
    case cell_kind::cable1d_neuron:
        if (group.backend == backend_kind::gpu) {
            return make_cell_group<gpu_fvm_cell>(group.gids, rec);
        }
        else {
            return make_cell_group<mc_fvm_cell>(group.gids, rec);
        }

    case cell_kind::regular_spike_source:
        return make_cell_group<rss_cell_group>(group.gids, rec);

    case cell_kind::lif_neuron:
        /*
        if (backend == backend_policy::prefer_gpu) {
            return make_cell_group<lif_cell_group_gpu>(first_gid, cell_descriptions);
        }
        else {
            return make_cell_group<lif_cell_group_mc>(first_gid, cell_descriptions);
        }
        */
// TODO (MK): change lif_cell_group_gpu class so that it accepts gids and rec instead of first_gid and cell_descriptions
#ifdef ARB_WITH_CUDA
        return make_cell_group<lif_cell_group_gpu>(first_gid, cell_descriptions);
#else
        return make_cell_group<lif_cell_group_mc>(group.gids, rec);
#endif

    case cell_kind::data_spike_source:
        return make_cell_group<dss_cell_group>(group.gids, rec);

    default:
        throw std::runtime_error("unknown cell kind");
    }
}

} // namespace arb
