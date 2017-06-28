#include "../gtest.h"

#include <vector>

#include <backends/gpu/kernels/reduce_by_key.hpp>
#include <memory/memory.hpp>
#include <util/span.hpp>
#include <util/rangeutil.hpp>

using namespace nest::mc;

template <typename T, typename I>
__global__
void reduce_kernel(const T* src, T* dst, const I* index, int n) {
    unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid<n) {
        gpu::reduce_by_key(src[tid], dst, index[tid]);
    }
}

template <typename T>
std::vector<T> reduce(const std::vector<T>& in, size_t n_out, const std::vector<int>& index, unsigned block_dim=128) {
    EXPECT_EQ(in.size(), index.size());
    EXPECT_TRUE(std::is_sorted(index.begin(), index.end()));

    using array = memory::device_vector<T>;
    using iarray = memory::device_vector<int>;

    int n = in.size();

    array  src = memory::make_const_view(in);
    iarray idx = memory::make_const_view(index);
    array  dst(n_out, 0);

    unsigned grid_dim = (n-1)/block_dim + 1;
    reduce_kernel<<<grid_dim, block_dim>>>(src.data(), dst.data(), idx.data(), n);

    std::vector<T> out(n_out);
    memory::copy(dst, memory::make_view(out));

    return out;
}

TEST(reduce_by_key, no_repetitions)
{
    int n = 64;
    {
        std::vector<float> in(n, 1);
        std::vector<int> index = util::assign_from(util::make_span(0, n));

        auto out = reduce(in, n, index);
        for (auto o: out) EXPECT_EQ(o, 1.0f);
    }
    {
        std::vector<double> in(n, 1);
        std::vector<int> index = util::assign_from(util::make_span(0, n));

        auto out = reduce(in, n, index);
        for (auto o: out) EXPECT_EQ(o, 1.0);
    }
}

TEST(reduce_by_key, single_repeated_index)
{
    // Perform reduction of a sequence of 1s of length n
    // The expected result is n
    for (auto n: {1, 2, 7, 31, 32, 33, 63, 64, 65, 128}) {
        std::vector<double> in(n, 1);
        std::vector<int> index(n, 0);

        auto out = reduce(in, 1, index);
        EXPECT_EQ(out[0], double(n));
    }
    // Perform reduction of an ascending sequence of {1,2,3,...,n}
    // The expected result is n*(n+1)/2
    for (auto n: {1, 2, 7, 31, 32, 33, 63, 64, 65, 128}) {
        std::vector<double> in = util::assign_from(util::make_span(1, n+1));
        std::vector<int> index(n, 0);

        auto out = reduce(in, 1, index);
        EXPECT_EQ(out[0], double((n+1)*n/2));
    }
}

TEST(reduce_by_key, scatter)
{
    std::vector<int> index = {0,0,0,1,2,2,3,7,7,7,11};
    unsigned n = util::max_value(index)+1;
    std::vector<double> in(index.size(), 1);
    std::vector<double> expected = {3., 1., 2., 1., 0., 0., 0., 3., 0., 0., 0., 1.};

    EXPECT_EQ(n, expected.size());

    auto out = reduce(in, n, index);
    EXPECT_EQ(out, expected);

    // rerun with 7 threads per thread block, to test
    //  * using more than one thread block
    //  * thread blocks that are not a multiple of 32
    //  * thread blocks that are less than 32
    out = reduce(in, n, index, 7);

    EXPECT_EQ(out, expected);
}
