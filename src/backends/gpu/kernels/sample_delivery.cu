#include <common_types.hpp>
#include <backends/event.hpp>
#include <backends/fvm_types.hpp>

namespace nest {
namespace mc {
namespace gpu {

namespace kernels {

    __global__ void run_samples(fvm_size_type n, fvm_value_type* store, const raw_probe_info* data, const fvm_size_type* begin, const fvm_size_type* end) {
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        if (i<n) {
            for (auto b = begin[i], e = end[i]; b!=e; ++b) {
                store[data[b].offset] = *data[b].handle;
            }
        }
    }

}

void run_samples(fvm_size_type n, fvm_value_type* store, const raw_probe_info* data, const fvm_size_type* begin, const fvm_size_type* end) {
    if (!n) {
        return;
    }

    constexpr int blockwidth = 128;
    int nblock = 1+(n-1)/blockwidth;
    kernels::run_samples<<<nblock, blockwidth>>>(n, store, data, begin, end);
}

} // namespace gpu
} // namespace mc
} // namespace nest

