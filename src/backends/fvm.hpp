#pragma once

#include "./multicore/fvm.hpp"

#ifdef NMC_HAVE_CUDA
    #include "./gpu/fvm.hpp"
#endif
