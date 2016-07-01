#pragma once

#if defined(WITH_TBB)
    #include "tbb.hpp"
#else
    #define WITH_SERIAL
    #include "serial.hpp"
#endif

