#include <algorithm>
#include <cstdint>
#include <type_traits>

//
// prototypes for compiled wrappers around fill kernels for GPU memory
//

namespace arb {
namespace gpu {

void fill8(uint8_t* v, uint8_t value, std::size_t n);
void fill16(uint16_t* v, uint16_t value, std::size_t n);
void fill32(uint32_t* v, uint32_t value, std::size_t n);
void fill64(uint64_t* v, uint64_t value, std::size_t n);

// Brief:
// Perform type punning to pass arbitrary POD types to the GPU backend
// without polluting the library front end with CUDA kernels that would
// require compilation with nvcc.
//
// Detail:
// The implementation takes advantage of 4 fill functions that fill GPU
// memory with a {8, 16, 32, 64} bit unsigned integer. These these functions
// are used to fill a block of GPU memory with _any_ 8, 16, 32 or 64 bit POD
// value. e.g. for a 64-bit double, first convert the double into a 64-bit
// unsigned integer (with the same bits, not the same value), then call the
// 64-bit fill kernel precompiled using nvcc in the gpu library. This
// technique of converting from one type to another is called type-punning.
// There are some subtle challenges, due to C++'s strict aliasing rules,
// that require memcpy of single bytes if alignment of the two types does
// not match.

#define FILL(N) \
template <typename T> \
std::enable_if_t<sizeof(T)==sizeof(uint ## N ## _t)> \
fill(T* ptr, T value, std::size_t n) { \
    using I = uint ## N ## _t; \
    I v; \
    std::copy_n( \
        reinterpret_cast<char*>(&value), \
        sizeof(T), \
        reinterpret_cast<char*>(&v) \
    ); \
    arb::gpu::fill ## N(reinterpret_cast<I*>(ptr), v, n); \
}

FILL(8)
FILL(16)
FILL(32)
FILL(64)

} // namespace gpu
} // namespace arb
