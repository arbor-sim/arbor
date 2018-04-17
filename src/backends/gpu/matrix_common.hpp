#pragma once

#include <cfloat>
#include <climits>

#ifdef __CUDACC__
#   define HOST_DEVICE_IF_CUDA __host__ __device__
#else
#   define HOST_DEVICE_IF_CUDA
#endif

namespace arb {
namespace gpu {

namespace impl {
// Number of matrices per block in block-interleaved storage
HOST_DEVICE_IF_CUDA
constexpr inline unsigned matrices_per_block() {
    return 32u;
}

// The number of threads per matrix in the interleave and reverse-interleave
// operations.
HOST_DEVICE_IF_CUDA
constexpr inline unsigned load_width() {
    return 32u;
}

// The alignment of matrices inside the block-interleaved storage.
HOST_DEVICE_IF_CUDA
constexpr inline unsigned matrix_padding() {
    return load_width();
}

// Placeholders to use for mark padded locations in data structures that use
// padding. Using such markers makes it easier to test that padding is
// performed correctly.
#define NPOS_SPEC(type, cint) template <> HOST_DEVICE_IF_CUDA constexpr type npos<type>() { return cint; }
template <typename T> HOST_DEVICE_IF_CUDA constexpr T npos();
NPOS_SPEC(char, CHAR_MAX)
NPOS_SPEC(unsigned char, UCHAR_MAX)
NPOS_SPEC(short, SHRT_MAX)
NPOS_SPEC(int, INT_MAX)
NPOS_SPEC(long, LONG_MAX)
NPOS_SPEC(float, FLT_MAX)
NPOS_SPEC(double, DBL_MAX)
NPOS_SPEC(unsigned short, USHRT_MAX)
NPOS_SPEC(unsigned int, UINT_MAX)
NPOS_SPEC(unsigned long, ULONG_MAX)
NPOS_SPEC(long long, LLONG_MAX)

// test if value v is npos
template <typename T>
HOST_DEVICE_IF_CUDA
constexpr bool is_npos(T v) {
    return v == npos<T>();
}

} // namespace impl

} // namespace gpu
} // namespace arb
