#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <dss_cell_group.hpp>
#include <fvm_multicell.hpp>
#include <lif_cell_group_mc.hpp>
#include <lif_cell_group_gpu.hpp>
#include <mc_cell_group.hpp>
#include <pss_cell_group.hpp>
#include <rss_cell_group.hpp>
#include <util/unique_any.hpp>

namespace nest {
namespace mc {

using gpu_fvm_cell = mc_cell_group<fvm::fvm_multicell<gpu::backend>>;
using mc_fvm_cell = mc_cell_group<fvm::fvm_multicell<multicore::backend>>;

cell_group_ptr cell_group_factory(
        cell_kind kind,
        cell_gid_type first_gid,
        const std::vector<util::unique_any>& cell_descriptions,
        backend_policy backend)
{
    switch (kind) {
    case cell_kind::cable1d_neuron:
        if (backend == backend_policy::prefer_gpu) {
            return make_cell_group<gpu_fvm_cell>(first_gid, cell_descriptions);
        }
        else {
            return make_cell_group<mc_fvm_cell>(first_gid, cell_descriptions);
        }

    case cell_kind::regular_spike_source:
        return make_cell_group<rss_cell_group>(first_gid, cell_descriptions);

    case cell_kind::lif_neuron:
        if (backend == backend_policy::prefer_gpu) {
            return make_cell_group<lif_cell_group_gpu>(first_gid, cell_descriptions);
        }
        else {
            return make_cell_group<lif_cell_group_mc>(first_gid, cell_descriptions);
        }

    case cell_kind::poisson_spike_source:
        return make_cell_group<pss_cell_group>(first_gid, cell_descriptions);

    case cell_kind::data_spike_source:
        return make_cell_group<dss_cell_group>(first_gid, cell_descriptions);

    default:
        throw std::runtime_error("unknown cell kind");
    }
}

} // namespace mc
} // namespace nest
