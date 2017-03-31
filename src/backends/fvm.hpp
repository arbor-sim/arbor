#pragma once

#include <backends/multicore/fvm.hpp>

#ifdef NMC_HAVE_CUDA
    #include <backends/gpu/fvm.hpp>
#endif
