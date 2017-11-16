// Override backend class in `test_spikes.cpp`

#include <backends/gpu/fvm.hpp>
#define USE_BACKEND gpu::backend

#include "./test_spikes.cpp"
