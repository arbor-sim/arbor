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

using gpu::impl::npos;
using util::make_span;
using util::assign_from;
using memory::on_gpu;
using memory::on_host;

using testing::seq_almost_eq;

using std::begin;
using std::end;

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

    auto in  = on_gpu(memory::make_const_view(values));
    auto sizes_d = on_gpu(memory::make_const_view(sizes));
    auto starts_d = on_gpu(memory::make_const_view(starts));

    int packed_size = padded_size * BlockWidth * gpu::impl::block_count(num_mtx, BlockWidth);

    // forward will hold the result of the interleave operation on the GPU
    auto forward = memory::device_vector<T>(packed_size, npos<T>());

    // find the reference interleaved values using host side implementation
    auto baseline = gpu::interleave_host(values, sizes, starts, BlockWidth, num_mtx, padded_size);

    // find the interleaved values on gpu
    gpu::interleave<T, I, BlockWidth, LoadWidth>(in.data(), forward.data(), sizes_d.data(), starts_d.data(), padded_size, num_mtx);

    std::vector<T> result_f = assign_from(on_host(forward));
    std::vector<T> expected = gpu::interleave_host(values, sizes, starts, BlockWidth, num_mtx, padded_size);
    const auto forward_success = (result_f==expected);
    EXPECT_TRUE(forward_success);

    // backward will hold the result of reverse interleave on the GPU
    auto backward = memory::device_vector<T>(values.size(), npos<T>());
    gpu::reverse_interleave<T, I, BlockWidth, LoadWidth>(forward.data(), backward.data(), sizes_d.data(), starts_d.data(), padded_size, num_mtx);

    std::vector<T> result_b = assign_from(on_host(backward));

    // we expect that the result of the reverse permutation is the original input vector
    const auto backward_success = (result_b==values);
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

    // There are two matrix structures:
    //
    // p_1: 3 branches, 6 compartments
    //
    //           3
    //          /.
    // 0 - 1 - 2
    //          \.
    //           4
    //            \.
    //             5
    //
    // p_2: 5 branches, 8 compartments
    //
    //             4
    //            /.
    //           3
    //          / \.
    // 0 - 1 - 2   5
    //          \.
    //           6
    //            \.
    //             7

    // The parent indexes that define the two matrix structures
    std::vector<std::vector<I>>
        p_base = { {0,0,1,2,2,4}, {0,0,1,2,3,3,2,6} };

    // Make a set of matrices based on repeating this pattern.
    // We assign the patterns round-robin, i.e. so that the input
    // matrices will have alternating sizes of 6 and 8, which will
    // test the solver with variable matrix size, and exercise
    // solvers that reorder matrices according to size.
    const int num_mtx = 8;

    std::vector<I> p;
    std::vector<I> cell_index;
    for (auto m=0; m<num_mtx; ++m) {
        auto &p_ref = p_base[m%2];
        auto first = p.size();
        for (auto i: p_ref) {
            p.push_back(i + first);
        }
        cell_index.push_back(first);
    }
    cell_index.push_back(p.size());

    auto group_size = cell_index.back();

    // Build the capacitance and conductance vectors and
    // populate with nonzero random values.

    auto gen  = std::mt19937();
    auto dist = std::uniform_real_distribution<T>(1, 2);

    std::vector<T> Cm(group_size);
    std::generate(Cm.begin(), Cm.end(), [&](){return dist(gen);});

    std::vector<T> g(group_size);
    std::generate(g.begin(), g.end(), [&](){return dist(gen);});

    // Make the referenace matrix and the gpu matrix
    auto m_mc  = mc_state( p, cell_index, Cm, g); // on host
    auto m_gpu = gpu_state(p, cell_index, Cm, g); // on gpu

    // Voltage and current values
    m_mc.assemble( 0.2, host_array(group_size, -64), host_array(group_size, 10));
    m_mc.solve();
    m_gpu.assemble(0.2, gpu_array(group_size, -64),  gpu_array(group_size, 10));
    m_gpu.solve();

    // Compare the GPU and CPU results.
    // Cast result to float, because we are happy to ignore small differencs
    EXPECT_TRUE(seq_almost_eq<float>(m_mc.solution, on_host(m_gpu.solution)));
}

// test that the flat and interleaved storage back ends produce identical results
TEST(matrix, backends)
{
    using T = typename gpu::backend::value_type;
    using I = typename gpu::backend::size_type;

    using state_flat = gpu::matrix_state_flat<T, I>;
    using state_intl = gpu::matrix_state_interleaved<T, I>;

    using gpu_array  = typename gpu::backend::array;

    // There are two matrix structures:
    //
    // p_1: 3 branches, 6 compartments
    //
    //           3
    //          /.
    // 0 - 1 - 2
    //          \.
    //           4
    //            \.
    //             5
    //
    // p_2: 5 branches, 8 compartments
    //
    //             4
    //            /.
    //           3
    //          / \.
    // 0 - 1 - 2   5
    //          \.
    //           6
    //            \.
    //             7

    // The parent indexes that define the two matrix structures
    std::vector<std::vector<I>>
        p_base = { {0,0,1,2,2,4}, {0,0,1,2,3,3,2,6} };

    // Make a set of matrices based on repeating this pattern.
    // We assign the patterns round-robin, i.e. so that the input
    // matrices will have alternating sizes of 6 and 8, which will
    // test the solver with variable matrix size, and exercise
    // solvers that reorder matrices according to size.
    const int num_mtx = 200;

    std::vector<I> p;
    std::vector<I> cell_index;
    for (auto m=0; m<num_mtx; ++m) {
        auto &p_ref = p_base[m%2];
        auto first = p.size();
        for (auto i: p_ref) {
            p.push_back(i + first);
        }
        cell_index.push_back(first);
    }
    cell_index.push_back(p.size());

    auto group_size = cell_index.back();

    // Build the capacitance and conductance vectors and
    // populate with nonzero random values

    auto gen  = std::mt19937();
    gen.seed(100);
    auto dist = std::uniform_real_distribution<T>(1, 200);

    std::vector<T> Cm(group_size);
    std::vector<T> g(group_size);
    std::vector<T> v(group_size);
    std::vector<T> i(group_size);

    std::generate(Cm.begin(), Cm.end(), [&](){return dist(gen);});
    std::generate(g.begin(), g.end(), [&](){return dist(gen);});
    std::generate(v.begin(), v.end(), [&](){return dist(gen);});
    std::generate(i.begin(), i.end(), [&](){return dist(gen);});

    // Make the referenace matrix and the gpu matrix
    auto flat = state_flat(p, cell_index, Cm, g); // flat
    auto intl = state_intl(p, cell_index, Cm, g); // interleaved

    // voltage and current values
    flat.assemble(0.02, on_gpu(v), on_gpu(i));
    intl.assemble(0.02, on_gpu(v), on_gpu(i));

    flat.solve();
    intl.solve();

    // Compare the results.
    // We expect exact equality for the two gpu matrix implementations because both
    // perform the same operations in the same order on the same inputs.
    std::vector<double> x_flat = assign_from(on_host(flat.solution));
    std::vector<double> x_intl = assign_from(on_host(intl.solution));
    EXPECT_EQ(x_flat, x_intl);
}
