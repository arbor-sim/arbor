#include <numeric>
#include <vector>

#include "../gtest.h"

#include <math.hpp>
#include <matrix.hpp>
#include <backends/fvm_multicore.hpp>
#include <util/span.hpp>

using namespace nest::mc;

using matrix_type = matrix<nest::mc::multicore::backend>;
using size_type = matrix_type::size_type;

TEST(matrix, construct_from_parent_only)
{
    std::vector<size_type> p = {0,0,1};
    matrix_type m(p, {0, 3}, {}, {});
    EXPECT_EQ(m.num_cells(), 1u);
    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(p.size(), 3u);

    auto mp = m.p();
    EXPECT_EQ(mp[0], 0u);
    EXPECT_EQ(mp[1], 0u);
    EXPECT_EQ(mp[2], 1u);
}

TEST(matrix, solve_host)
{
    using util::make_span;
    using memory::fill;

    // trivial case : 1x1 matrix
    {
        matrix_type m({0}, {0,1}, {}, {});
        auto& state = m.state_;
        fill(state.d,  2);
        fill(state.u, -1);
        fill(state.rhs,1);

        m.solve();

        EXPECT_EQ(m.solution()[0], 0.5);
    }

    // matrices in the range of 2x2 to 1000x1000
    {
        for(auto n : make_span(2u,1001u)) {
            auto p = std::vector<size_type>(n);
            std::iota(p.begin()+1, p.end(), 0);
            matrix_type m(p, {0, n}, {}, {});

            EXPECT_EQ(m.size(), n);
            EXPECT_EQ(m.num_cells(), 1u);

            auto& A = m.state_;

            fill(A.d,  2);
            fill(A.u, -1);
            fill(A.rhs,1);

            m.solve();

            auto x = m.solution();
            auto err = math::square(std::fabs(2.*x[0] - x[1] - 1.));
            for(auto i : make_span(1,n-1)) {
                err += math::square(std::fabs(2.*x[i] - x[i-1] - x[i+1] - 1.));
            }
            err += math::square(std::fabs(2.*x[n-1] - x[n-2] - 1.));

            EXPECT_NEAR(0., std::sqrt(err), 1e-8);
        }
    }
}
