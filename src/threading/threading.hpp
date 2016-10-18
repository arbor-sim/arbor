#pragma once

#if defined(WITH_TBB)
    #include "tbb.hpp"
#elif defined(WITH_OMP)
    #include "omp.hpp"
#else
    #define WITH_SERIAL
    #include "serial.hpp"
#endif

