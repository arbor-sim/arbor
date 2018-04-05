#include <memory>
#include <stdexcept>

#include <backends.hpp>
#include <backends/multicore/fvm.hpp>
#ifdef ARB_HAVE_GPU
#include <backends/gpu/fvm.hpp>
#endif
#include <fvm_lowered_cell_impl.hpp>

namespace arb {

fvm_lowered_cell_ptr make_fvm_lowered_cell(backend_kind p) {
    switch (p) {
    case backend_kind::multicore:
        return fvm_lowered_cell_ptr(new fvm_lowered_cell_impl<multicore::backend>);
    case backend_kind::gpu:
#ifdef ARB_HAVE_GPU
        return fvm_lowered_cell_ptr(new fvm_lowered_cell_impl<multicore::backend>);
#endif
        ; // fall through
    default:
        throw std::logic_error("unsupported back-end");
    }
}

} // namespace arb
