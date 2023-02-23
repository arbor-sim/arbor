#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <arbor/math.hpp>

#include "backends/multicore/fvm.hpp"
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

TEST(matrix, construct_from_parent_only)
{
    std::vector<index_type> p = {0,0,1};
    solver_type m(p, {0, 3}, vvec(3), vvec(3), vvec(3), {0});
    EXPECT_EQ(m.num_cells(), 1u);
    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(p.size(), 3u);

    auto mp = m.parent_index;
    EXPECT_EQ(mp[0], index_type(0));
    EXPECT_EQ(mp[1], index_type(0));
    EXPECT_EQ(mp[2], index_type(1));
}

TEST(matrix, solve_host)
{
    using util::make_span;
    using util::fill;

    // trivial case : 1x1 matrix
    {
        solver_type m({0}, {0,1}, vvec(1), vvec(1), vvec(1), {0});
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
            solver_type m(p, {0, n}, vvec(n), vvec(n), vvec(n), {0});

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

TEST(matrix, zero_diagonal)
{
    // Combined matrix may have zero-blocks, corresponding to a zero dt.
    // Zero-blocks are indicated by zero value in the diagonal (the off-diagonal
    // elements should be ignored).
    // These submatrices should leave the rhs as-is when solved.

    using util::assign;

    // Three matrices, sizes 3, 3 and 2, with no branching.
    std::vector<index_type> p = {0, 0, 1, 3, 3, 5, 5};
    std::vector<index_type> c = {0, 3, 5, 7};
    std::vector<index_type> i = {0, 1, 2};
    solver_type m(p, c, vvec(7), vvec(7), vvec(7), i);

    EXPECT_EQ(7u, m.size());
    EXPECT_EQ(3u, m.num_cells());

    assign(m.d,   vvec({2,  3,  2, 0,  0,  4,  5}));
    assign(m.u,   vvec({0, -1, -1, 0, -1,  0, -2}));

    // Expected solution:
    std::vector<value_type> expected = {4, 5, 6, 7, 8, 9, 10};

    auto x = vvec({3,  5,  7, 7,  8, 16, 32});
    m.solve(x);

    EXPECT_TRUE(testing::seq_almost_eq<double>(expected, x));
}

TEST(matrix, zero_diagonal_assembled)
{
    // Use assemble method to construct same zero-diagonal
    // test case from CV data.

    using util::assign;
    using array = solver_type::array;

    // Combined matrix may have zero-blocks, corresponding to a zero dt.
    // Zero-blocks are indicated by zero value in the diagonal (the off-diagonal
    // elements should be ignored).
    // These submatrices should leave the rhs as-is when solved.

    // Three matrices, sizes 3, 3 and 2, with no branching.
    std::vector<index_type> p = {0, 0, 1, 3, 3, 5, 5};
    std::vector<index_type> c = {0, 3, 5, 7};
    std::vector<index_type> s = {0, 1, 2};

    // Face conductances.
    vvec g = {0, 1, 1, 0, 1, 0, 2};

    // dt of 1e-3.
    array dt(3, 1.0e-3);

    // Capacitances.
    vvec Cm = {1, 1, 1, 1, 1, 2, 3};

    // Initial voltage of zero; currents alone determine rhs.
    auto v = vvec{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    vvec area(7, 1.0);

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

    solver_type m(p, c, Cm, g, area, s);
    std::vector<value_type> expected = {4, 5, 6, 7, 8, 9, 10};

    m.solve(v, dt, i, mg);
    EXPECT_TRUE(testing::seq_almost_eq<double>(expected, v));

    // Set dt of 2nd (middle) submatrix to zero. Solution
    // should then return voltage values for that submatrix.

    dt[1] = 0;
    v = vvec{0.0, 0.0, 0.0, -20.0, -30.0, 0.0, 0.0};
    m.solve(v, dt, i, mg);
    expected = {4, 5, 6, -20, -30, 9, 10};

    EXPECT_TRUE(testing::seq_almost_eq<double>(expected, v));
}
