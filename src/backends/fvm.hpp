#pragma once

#include <backends/multicore/fvm.hpp>

#ifdef NMC_HAVE_CUDA
    #include <backends/gpu/fvm.hpp>
#else
// FIXME: intermediate fix as part of virtualization of cell_group.
// This requires that a runtime choice can be made during model set up
// about which backend is to be used group by group. We use the multicore
// backend as a dummy GPU back end when compiled without gpu support so that
// this decision does not require preprocessor directives in model.hpp.
namespace nest { namespace mc { namespace gpu {
    using backend = nest::mc::multicore::backend;
}}} // namespace nest::nc::gpu

#endif
