#pragma once

// Decorator adds '__host__ __device__' to methods/functions when compiled with nvcc.

#ifdef __CUDACC__
#define CUDA_DECORATE __host__ __device__
#else
#define CUDA_DECORATE
#endif
