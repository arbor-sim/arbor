#pragma once

#include <cstdint>
#include <cfloat>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace nest {
namespace mc {
namespace gpu {

namespace impl {
// Number of matrices per block in block-interleaved storage
__host__ __device__
constexpr inline int block_dim() {
    return 32;
}

// The number of threads per matrix in the interleave and reverse-interleave
// operations.
__host__ __device__
constexpr inline int load_width() {
    return 32;
}

// The alignment of matrices inside the block-interleaved storage.
__host__ __device__
constexpr inline int matrix_padding() {
    return load_width();
}

// Number of threads per warp
// This has always been 32, however it may change in future NVIDIA gpus
__host__ __device__
constexpr inline int threads_per_warp() {
    return 32;
}

// The minimum number of bins required to store n values where the bins have
// dimension of block_size.
__host__ __device__
constexpr inline int block_count(int n, int block_size) {
    return (n+block_size-1)/block_size;
}

// The smallest size of a buffer required to store n items in such that the
// buffer has size that is a multiple of block_dim.
constexpr inline int padded_size (int n, int block_dim) {
    return n%block_dim ? n+block_dim-(n%block_dim): n;
}

// Placeholders to use for mark padded locations in data structures that use
// padding. Using such markers makes it easier to test that padding is
// performed correctly.
template <typename T> __host__ __device__ constexpr T npos();
template <> __host__ __device__ constexpr char npos<char>() { return CHAR_MAX; }
template <> __host__ __device__ constexpr unsigned char npos<unsigned char>() { return UCHAR_MAX; }
template <> __host__ __device__ constexpr short npos<short>() { return SHRT_MAX; }
template <> __host__ __device__ constexpr int npos<int>() { return INT_MAX; }
template <> __host__ __device__ constexpr long npos<long>() { return LONG_MAX; }
template <> __host__ __device__ constexpr float npos<float>() { return FLT_MAX; }
template <> __host__ __device__ constexpr double npos<double>() { return DBL_MAX; }
template <> __host__ __device__ constexpr unsigned short npos<unsigned short>() { return USHRT_MAX; }
template <> __host__ __device__ constexpr unsigned int npos<unsigned int>() { return UINT_MAX; }
template <> __host__ __device__ constexpr unsigned long npos<unsigned long>() { return ULONG_MAX; }
template <> __host__ __device__ constexpr long long npos<long long>() { return LLONG_MAX; }

// test if value v is npos
template <typename T>
__host__ __device__
constexpr bool is_npos(T v) {
    return v == npos<T>();
}

} // namespace impl

} // namespace gpu
} // namespace mc
} // namespace nest
