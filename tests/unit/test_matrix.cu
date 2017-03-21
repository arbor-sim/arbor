#include <numeric>
#include <vector>

#include "../gtest.h"

#include <math.hpp>
#include <matrix.hpp>
#include <backends/fvm_gpu.hpp>
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

template <typename T, typename I, int BlockWidth, int LoadWidth>
void test_interleave(std::vector<I> sizes, std::vector<I> starts, std::vector<T> values, int padded_size) {
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

    if (!forward_success) {
        //print_vec("result_f", result_f);
        //print_vec("expected", expected);
    }
    EXPECT_TRUE(forward_success);

    // backward will hold the result of reverse interleave on the GPU
    auto backward = memory::device_vector<T>(values.size(), npos<T>());
    gpu::reverse_interleave<T, I, BlockWidth, LoadWidth>(forward.data(), backward.data(), sizes_d.data(), starts_d.data(), padded_size, num_mtx);

    std::vector<T> result_b = util::assign_from(memory::on_host(backward));

    // we expect that the result of the reverse permutation is the original input vector
    const auto backward_success = (result_b==values);
    if (!backward_success) {
        print_vec("result_b", result_b);
        print_vec("expected", values);
    }
    //std::cout << BlockWidth << " " << LoadWidth << " : " << (backward_success?"good":"bad") << "\n";
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

TEST(matrix, solve_gpu)
{
    using namespace nest::mc;

    using nest::mc::util::make_span;

    // trivial case : 1x1 matrix
    {
        matrix_type m({0}, {0,1}, {}, {});

        auto& state = m.state_;
        memory::fill(state.d,  2);
        memory::fill(state.u, -1);
        memory::fill(state.rhs,1);

        m.solve();

        auto rhs = memory::on_host(m.solution());

        EXPECT_EQ(rhs[0], 0.5);
    }

    // matrices in the range of 2x2 to 100x100
    /*
    {
        using namespace nest::mc;
        for(auto n : make_span(2u,101u)) {
            auto p = std::vector<index_type>(n);
            std::iota(p.begin()+1, p.end(), 0);
            matrix_type m{p, {0, n}, {}, {}};

            EXPECT_EQ(m.size(), n);
            EXPECT_EQ(m.num_cells(), 1u);

            auto& state = m.state_;
            memory::fill(state.d,  2);
            memory::fill(state.u, -1);
            memory::fill(state.rhs,1);

            m.solve();

            auto x = memory::on_host(m.solution());
            auto err = math::square(std::fabs(2.*x[0] - x[1] - 1.));
            for(auto i : make_span(1,n-1)) {
                err += math::square(std::fabs(2.*x[i] - x[i-1] - x[i+1] - 1.));
            }
            err += math::square(std::fabs(2.*x[n-1] - x[n-2] - 1.));

            EXPECT_NEAR(0., std::sqrt(err), 1e-8);
        }
    }
    */
}
