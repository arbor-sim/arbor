#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <fvm_multicell.hpp>
#include <mc_cell_group.hpp>
#include <util/unique_any.hpp>

namespace nest {
namespace mc {

using gpu_fvm_cell = mc_cell_group<fvm::fvm_multicell<gpu::backend>>;
using mc_fvm_cell = mc_cell_group<fvm::fvm_multicell<multicore::backend>>;

cell_group_ptr cell_group_factory(
        cell_kind kind,
        cell_gid_type first_gid,
        const std::vector<util::unique_any>& cells,
        backend_policy backend)
{
    if (backend==backend_policy::prefer_gpu) {
        switch (kind) {
        case cell_kind::cable1d_neuron:
            return make_cell_group<gpu_fvm_cell>(first_gid, cells);
        default:
            throw std::runtime_error("unknown cell kind");
        }
    }

    switch (kind) {
    case cell_kind::cable1d_neuron:
        return make_cell_group<mc_fvm_cell>(first_gid, cells);
    default:
        throw std::runtime_error("unknown cell kind");
    }
}

} // namespace mc
} // namespace nest
