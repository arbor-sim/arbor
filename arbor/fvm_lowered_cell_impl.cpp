#include <memory>
#include <stdexcept>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/version.hpp>

#include "backends/multicore/fvm.hpp"
#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#endif
#include "fvm_lowered_cell_impl.hpp"

namespace arb {

fvm_lowered_cell_ptr make_fvm_lowered_cell(backend_kind p, const execution_context& ctx,
                                           arb_seed_type seed) {
    switch (p) {
    case backend_kind::multicore:
        return fvm_lowered_cell_ptr(new fvm_lowered_cell_impl<multicore::backend>(ctx, seed));
    case backend_kind::gpu:
#ifdef ARB_GPU_ENABLED
        return fvm_lowered_cell_ptr(new fvm_lowered_cell_impl<gpu::backend>(ctx, seed));
#endif
        ; // fall through
    default:
        throw arbor_internal_error("fvm_lowered_cell: unsupported back-end");
    }
}

} // namespace arb
