#include <ostream>

#include <cuda_runtime.h>

#include "memory/cuda_wrappers.hpp"
#include "util/span.hpp"

#include "matrix_fine.hpp"

namespace arb {
namespace gpu {

level::level(unsigned branches):
    num_branches(branches)
{
    using memory::cuda_malloc_managed;

    using arb::memory::cuda_malloc_managed;
    if (num_branches!=0) {
        lengths = static_cast<unsigned*>(cuda_malloc_managed(num_branches*sizeof(unsigned)));
        parents = static_cast<unsigned*>(cuda_malloc_managed(num_branches*sizeof(unsigned)));
        cudaDeviceSynchronize();
    }
}

level::level(level&& other) {
    std::swap(other.lengths, this->lengths);
    std::swap(other.parents, this->parents);
    std::swap(other.num_branches, this->num_branches);
    std::swap(other.max_length, this->max_length);
    std::swap(other.data_index, this->data_index);
}

level::level(const level& other) {
    using memory::cuda_malloc_managed;

    num_branches = other.num_branches;
    max_length = other.max_length;
    data_index = other.data_index;
    if (num_branches!=0) {
        lengths = static_cast<unsigned*>(cuda_malloc_managed(num_branches*sizeof(unsigned)));
        parents = static_cast<unsigned*>(cuda_malloc_managed(num_branches*sizeof(unsigned)));
        cudaDeviceSynchronize();
        std::copy(other.lengths, other.lengths+num_branches, lengths);
        std::copy(other.parents, other.parents+num_branches, parents);
    }
}

level::~level() {
    if (num_branches!=0) {
        cudaDeviceSynchronize(); // to ensure that managed memory has been freed
        if (lengths) arb::memory::cuda_free(lengths);
        if (parents) arb::memory::cuda_free(parents);
    }
}

std::ostream& operator<<(std::ostream& o, const level& l) {
    cudaDeviceSynchronize();
    o << "branches:" << l.num_branches
      << " max_len:" << l.max_length
      << " data_idx:" << l.data_index
      << " lengths:[";
    for (auto i: util::make_span(l.num_branches)) o << l.lengths[i] << " ";
    o << "] parents:[";
    for (auto i: util::make_span(l.num_branches)) o << l.parents[i] << " ";
    o << "]";
    return o;
}

} // namespace gpu
} // namespace arb
