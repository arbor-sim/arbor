#include <numeric>
#include <vector>

#include "../gtest.h"
#include "common.hpp"

#include <math.hpp>
#include <matrix.hpp>
#include <backends/fvm_gpu.hpp>
#include <backends/fvm_multicore.hpp>
#include <memory/memory.hpp>
#include <util/span.hpp>

#include <backends/gpu_kernels.hpp>

using namespace nest::mc;

using matrix_type = nest::mc::matrix<nest::mc::gpu::backend>;
using index_type = matrix_type::size_type;

using std::begin;
using std::end;

using gpu::impl::print_vec;
using gpu::impl::npos;

using util::make_span;

using testing::seq_almost_eq;

template <typename T>
bool operator ==(const std::vector<T>& l, const std::vector<T>& r) {
    if (l.size()!=r.size()) {
        return false;
    }

    for (auto i=0u; i<l.size(); ++i) {
        if (l[i]!=r[i]) {
            return false;
        }
    }

    return true;
}
template <typename T>
bool operator !=(const std::vector<T>& l, const std::vector<T>& r) {
    return !(l==r);
}

// will test the interleaving and reverse_interleaving operations for the
// set of matrices defined by sizes and starts.
// Applies the interleave to the vector in values, and checks this against
// a reference result generated using a host side reference implementation.
// Then the interleave result is reverse_interleaved, and the result is
// compared to the original input.
//
// This is implemented in a separate function to facilitate testing on a
// broad range of BlockWidth and LoadWidth compile time parameters.
template <typename T, typename I, int BlockWidth, int LoadWidth>
void test_interleave(
        std::vector<I> sizes,
        std::vector<I> starts,
        std::vector<T> values,
        int padded_size)
{
    auto num_mtx = sizes.size();

    auto in  = memory::on_gpu(memory::make_const_view(values));
    auto sizes_d = memory::on_gpu(memory::make_const_view(sizes));
    auto starts_d = memory::on_gpu(memory::make_const_view(starts));

    int packed_size = padded_size * BlockWidth * gpu::impl::block_count(num_mtx, BlockWidth);

    // forward will hold the result of the interleave operation on the GPU
    auto forward = memory::device_vector<T>(packed_size, npos<T>());

    auto baseline = gpu::interleave_host(values, sizes, starts, BlockWidth, num_mtx, padded_size);

    // template parameters: T I BlockWidth load_width
    gpu::interleave<T, I, BlockWidth, LoadWidth>(in.data(), forward.data(), sizes_d.data(), starts_d.data(), padded_size, num_mtx);

    std::vector<T> result_f = util::assign_from(memory::on_host(forward));
    std::vector<T> expected = gpu::interleave_host(values, sizes, starts, BlockWidth, num_mtx, padded_size);
    const auto forward_success = (result_f==expected);

    //if (!forward_success) {
    //    print_vec("result_f", result_f);
    //    print_vec("expected", expected);
    //}
    EXPECT_TRUE(forward_success);

    // backward will hold the result of reverse interleave on the GPU
    auto backward = memory::device_vector<T>(values.size(), npos<T>());
    gpu::reverse_interleave<T, I, BlockWidth, LoadWidth>(forward.data(), backward.data(), sizes_d.data(), starts_d.data(), padded_size, num_mtx);

    std::vector<T> result_b = util::assign_from(memory::on_host(backward));

    // we expect that the result of the reverse permutation is the original input vector
    const auto backward_success = (result_b==values);
    //if (!backward_success) {
    //    print_vec("result_b", result_b);
    //    print_vec("expected", values);
    //}
    EXPECT_TRUE(backward_success);
}

// test conversion to and from interleaved back end storage format
TEST(matrix, interleave)
{
    using I = int;
    using T = int;
    using ivec = std::vector<I>;
    using tvec = std::vector<T>;

    // simple case with 4 matrices of length 2
    {
        const int padded_size = 2;
        const int num_mtx = 4;
        ivec sizes(num_mtx, padded_size);

        // find the start position of each matrix in the flat storage
        // we are assuming that the matrices are unpermuted
        ivec starts(num_mtx, 0);
        std::partial_sum(begin(sizes), end(sizes)-1, begin(starts)+1);

        tvec values(padded_size*num_mtx);
        std::iota(values.begin(), values.end(), 0);

        test_interleave<T, I, 1, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 2, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 3, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 4, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 5, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 6, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 7, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 8, 1>(sizes, starts, values, padded_size);

        test_interleave<T, I, 1, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 2, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 3, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 4, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 5, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 6, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 7, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 8, 2>(sizes, starts, values, padded_size);

        test_interleave<T, I, 1, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 2, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 3, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 4, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 5, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 6, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 7, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 8, 3>(sizes, starts, values, padded_size);
    }

    // another small example with matrices of differing lengths
    {
        const int padded_size = 8;
        const int num_mtx = 8;
        ivec sizes = {6, 5, 4, 4, 3, 2, 2, 1};

        // find the start position of each matrix in the flat storage
        // we are assuming that the matrices are unpermuted
        ivec starts(num_mtx, 0);
        std::partial_sum(begin(sizes), end(sizes)-1, begin(starts)+1);

        tvec values(algorithms::sum(sizes));
        std::iota(values.begin(), values.end(), 0);

        test_interleave<T, I, 1, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 2, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 3, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 4, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 5, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 6, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 7, 1>(sizes, starts, values, padded_size);
        test_interleave<T, I, 8, 1>(sizes, starts, values, padded_size);

        test_interleave<T, I, 1, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 2, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 3, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 4, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 5, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 6, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 7, 2>(sizes, starts, values, padded_size);
        test_interleave<T, I, 8, 2>(sizes, starts, values, padded_size);

        test_interleave<T, I, 1, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 2, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 3, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 4, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 5, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 6, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 7, 3>(sizes, starts, values, padded_size);
        test_interleave<T, I, 8, 3>(sizes, starts, values, padded_size);
    }

    // more interesting case...
    {
        const int padded_size = 256;
        const int num_mtx = 1000;
        ivec sizes(num_mtx);
        for (auto i: make_span(  0, 100)) sizes[i] = 250;
        for (auto i: make_span(100, 103)) sizes[i] = 213;
        for (auto i: make_span(103, 150)) sizes[i] = 200;
        for (auto i: make_span(150, 500)) sizes[i] = 178;
        for (auto i: make_span(500, 999)) sizes[i] = 6;

        // we are assuming that the matrices are unpermuted
        ivec starts(num_mtx, 0);
        std::partial_sum(begin(sizes), end(sizes)-1, begin(starts)+1);

        tvec values(algorithms::sum(sizes));
        std::iota(values.begin(), values.end(), 0);

        // test in "full" 1024 thread configuration with 32 threads per matrix
        test_interleave<T, I, 32, 32>(sizes, starts, values, padded_size);
    }
}

// Test that matrix assembly works.
// The test proceeds by assembling a reference matrix on the host and
// device backends, then performs solve, and compares solution.
//
// limitations of test
//  * matrices all have same size and structure
TEST(matrix, assemble)
{
    using gpu_state = gpu::backend::matrix_state;
    using mc_state  = multicore::backend::matrix_state;

    using T = typename gpu::backend::value_type;
    using I = typename gpu::backend::size_type;

    using gpu_array  = typename gpu::backend::array;
    using host_array = typename multicore::backend::array;

    // single cell has the following structure
    //           3
    //          /.
    // 0 - 1 - 2
    //          \.
    //           4
    //            \.
    //             5
    // which is has the following parent index
    std::vector<I> p_base = {0, 0, 1, 2, 2, 4};

    // make a set of matrices based on repeating this pattern
    const int num_mtx = 8;
    const int mtx_size = p_base.size();

    std::vector<I> p;
    for (auto m=0; m<num_mtx; ++m) {
        for (auto i: p_base) {
            p.push_back(i + m*mtx_size);
        }
    }

    const int group_size = p.size();

    std::vector<I> cell_index;
    for (auto i=0; i<num_mtx+1; ++i) {
        cell_index.push_back(i*mtx_size);
    }

    // build the capacitance and conductance vectors and
    // populate with nonzero random values

    auto gen  = std::mt19937();
    auto dist = std::uniform_real_distribution<T>(1, 2);

    std::vector<T> Cm(group_size);
    std::generate(Cm.begin(), Cm.end(), [&](){return dist(gen);});

    std::vector<T> g(group_size);
    std::generate(g.begin(), g.end(), [&](){return dist(gen);});

    // make the referenace matrix and the gpu matrix
    auto m_mc  = mc_state( p, cell_index, Cm, g); // on host
    auto m_gpu = gpu_state(p, cell_index, Cm, g); // on gpu

    // voltage and current values
    m_mc.assemble( 0.2, host_array(group_size, -64), host_array(group_size, 10));
    m_mc.solve();
    m_gpu.assemble(0.2, gpu_array(group_size, -64),  gpu_array(group_size, 10));
    m_gpu.solve();

    // Inspect the results.
    // Cast result to float, because we are happy to ignore small differencs
    // in the results. TODO: implement an EXPECT_NEAR for sequences
    EXPECT_TRUE(seq_almost_eq<float>(m_mc.solution, memory::on_host(m_gpu.solution)));
}

