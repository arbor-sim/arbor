#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <arbor/math.hpp>

#include "backends/multicore/fvm.hpp"
#ifdef ARB_GPU_ENABLED
#include "backends/gpu/matrix_fine.hpp"
#include "memory/wrappers.hpp"
#endif
#include "util/rangeutil.hpp"
#include "util/span.hpp"

#include "common.hpp"

using namespace arb;

using backend     = multicore::backend;
using array       = backend::array;
using solver_type = backend::cable_solver;
using index_type  = arb_index_type;
using value_type  = arb_value_type;

using vvec = std::vector<value_type>;

#ifdef ARB_GPU_ENABLED
TEST(matrix, gpu_fixed_parent_accumulation) {
    // Three trees per block, with children on both ends of each root branch.
    // One tree is inactive (zero diagonal); it must retain its original RHS.
    constexpr unsigned width = 96, length = 8, roots = 3, blocks = 32;
    constexpr unsigned size = width*length + 2*roots;
    std::vector<double> diagonal(size*blocks), rhs(diagonal.size()), upper(diagonal.size());
    std::vector<int> parent(diagonal.size(), -1), lengths((width+roots)*blocks, 2);
    std::vector<int> level_parents(lengths.size()), block_index(blocks+1), count(blocks, roots);
    std::vector<gpu::level_metadata> levels(2*blocks);
    for (unsigned b = 0; b<blocks; ++b) {
        block_index[b] = 2*b;
        levels[2*b] = {width, length, b*size, b*(width+roots)};
        levels[2*b+1] = {roots, 2, b*size+width*length, b*(width+roots)+width};
        for (unsigned i = 0; i<width; ++i) {
            unsigned len = 1+(17*i+3)%length;
            lengths[b*(width+roots)+i] = len;
            level_parents[b*(width+roots)+i] = i%(2*roots);
            for (unsigned k = 0; k<len; ++k) {
                unsigned p = b*size+k*width+i;
                diagonal[p] = i%roots==1? 0: 2+0.01*i+0.02*k;
                rhs[p] = 0.1+std::sin(double(i+k));
                upper[p] = -0.1;
                parent[p] = k+1<len? p+width: b*size+width*length+i%(2*roots);
            }
        }
        for (unsigned i = 0; i<2*roots; ++i) {
            unsigned p = b*size+width*length+i;
            diagonal[p] = i%roots==1? 0: 100;
            rhs[p] = 1;
            if (i<roots) { parent[p] = p+roots; upper[p] = -0.1; }
        }
    }
    block_index[blocks] = 2*blocks;
    auto u = memory::on_gpu(upper);
    auto meta = memory::on_gpu(levels);
    auto lens = memory::on_gpu(lengths);
    auto parents = memory::on_gpu(level_parents);
    auto block = memory::on_gpu(block_index);
    auto cells = memory::on_gpu(count);
#ifdef ARB_DETERMINISTIC_GPU
    std::vector<double> first;
#endif
    for (unsigned repeat = 0; repeat<100; ++repeat) {
        auto d = memory::on_gpu(diagonal);
        auto x = memory::on_gpu(rhs);
        gpu::solve_matrix_fine(x.data(), d.data(), u.data(), meta.data(), lens.data(),
                               parents.data(), block.data(), cells.data(), nullptr, blocks, 128);
        auto host = memory::on_host(x);
        std::vector<double> actual(host.begin(), host.end());
#ifdef ARB_DETERMINISTIC_GPU
        if (!repeat) first = actual;
        EXPECT_EQ(first, actual);
#endif
        if (repeat) continue;
        std::vector<double> residual(rhs.size());
        for (unsigned i = 0; i<rhs.size(); ++i) {
            if (!diagonal[i]) { EXPECT_EQ(rhs[i], actual[i]); continue; }
            residual[i] += diagonal[i]*actual[i]-rhs[i];
            if (parent[i]>=0) {
                residual[i] += upper[i]*actual[parent[i]];
                residual[parent[i]] += upper[i]*actual[i];
            }
        }
        for (auto r: residual) EXPECT_NEAR(0, r, 1e-12);
    }
}
#endif

TEST(matrix, construct_from_parent_only) {
    std::vector<index_type> p = {0,0,1};
    solver_type m(p, {0, 3}, vvec(3), vvec(3));
    EXPECT_EQ(m.num_cells(), 1u);
    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(p.size(), 3u);

    auto mp = m.parent_index;
    EXPECT_EQ(mp[0], index_type(0));
    EXPECT_EQ(mp[1], index_type(0));
    EXPECT_EQ(mp[2], index_type(1));
}

TEST(matrix, solve_host) {
    using util::make_span;
    using util::fill;

    // trivial case : 1x1 matrix
    {
        solver_type m({0}, {0,1}, vvec(1), vvec(1));
        fill(m.d,  2);
        fill(m.u, -1);
        array x({1});
        m.solve(x);

        EXPECT_EQ(x[0], 0.5);
    }

    // matrices in the range of 2x2 to 1000x1000
    {
        for(auto n : make_span(2, 1001)) {
            auto p = std::vector<index_type>(n);
            std::iota(p.begin()+1, p.end(), 0);
            solver_type m(p, {0, n}, vvec(n), vvec(n));

            EXPECT_EQ(m.size(), (unsigned)n);
            EXPECT_EQ(m.num_cells(), 1u);

            fill(m.d,  2);
            fill(m.u, -1);
            auto x = array(n, 1);

            m.solve(x);

            auto err = math::square(std::fabs(2.*x[0] - x[1] - 1.));
            for(auto i : make_span(1,n-1)) {
                err += math::square(std::fabs(2.*x[i] - x[i-1] - x[i+1] - 1.));
            }
            err += math::square(std::fabs(2.*x[n-1] - x[n-2] - 1.));

            EXPECT_NEAR(0., std::sqrt(err), 1e-8);
        }
    }
}

TEST(matrix, solve_multi_matrix) {
    // Use assemble method to construct same zero-diagonal
    // test case from CV data.

    using util::assign;
    using array = solver_type::array;

    // Combined matrix may have zero-blocks, corresponding to a zero dt.
    // Zero-blocks are indicated by zero value in the diagonal (the off-diagonal
    // elements should be ignored).
    // These submatrices should leave the rhs as-is when solved.

    // Three matrices, sizes 3, 3 and 2, with no branching.
    // parent index:
    std::vector<index_type> p = {0, 0, 1, 3, 3, 5, 5};
    // cell divisipn index:
    std::vector<index_type> c = {0, 3, 5, 7};

    value_type dt = 1e-3;

    // Face conductances.
    vvec g = {0, 1, 1, 0, 1, 0, 2};

    // Capacitances.
    vvec Cm = {1, 1, 1, 1, 1, 2, 3};

    // Initial voltage of zero; currents alone determine rhs.
    auto v = vvec{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    array area(7, 1.0);

    // (Scaled) membrane conductances contribute to diagonal.
    array mg = { 1000, 2000, 3000, 4000, 5000, 6000, 7000 };
    array i = {-7000, -15000, -25000, -34000, -49000, -70000, -102000};

    // Expected matrix and rhs:
    // u   = [ 0 -1 -1  0 -1  0 -2]
    // d   = [ 3  5  5  6  7  10  12]
    // rhs = [ 7 15 25 34 49 70 102 ]
    //
    // Expected solution:
    // x = [ 4 5 6 7 8 9 10 ]

    solver_type m(p, c, Cm, g);
    std::vector<value_type> expected = {4, 5, 6, 7, 8, 9, 10};

    m.solve(v, dt, i, mg, area);
    EXPECT_TRUE(testing::seq_almost_eq<double>(expected, v));
}
