#include <numeric>
#include <vector>

#include "gtest.h"

#include <math.hpp>
#include <matrix.hpp>
#include <util/span.hpp>

TEST(matrix, construct_from_parent_only)
{
    using nest::mc::util::make_span;
    using matrix_type = nest::mc::matrix<double, int, nest::mc::multicore::matrix_policy>;

    // pass parent index as a std::vector cast to host data
    {
        std::vector<int> p = {0,0,1};
        matrix_type m(memory::on_host(p));
        EXPECT_EQ(m.num_cells(), 1);
        EXPECT_EQ(m.size(), 3u);
        EXPECT_EQ(p.size(), 3u);

        auto mp = m.p();
        EXPECT_EQ(mp[0], 0);
        EXPECT_EQ(mp[1], 0);
        EXPECT_EQ(mp[2], 1);
    }
}

TEST(matrix, solve_host)
{
    using nest::mc::util::make_span;
    using matrix_type = nest::mc::matrix<double, int, nest::mc::multicore::matrix_policy>;

    // trivial case : 1x1 matrix
    {
        matrix_type m(memory::on_host(std::vector<int>{0}));

        memory::fill(m.d(),  2);
        memory::fill(m.l(), -1);
        memory::fill(m.u(), -1);
        memory::fill(m.rhs(),1);

        m.solve();

        EXPECT_EQ(m.rhs()[0], 0.5);
    }
    // matrices in the range of 2x2 to 1000x1000
    {
        using namespace nest::mc;
        for(auto n : make_span(2u,1001u)) {
            auto p = std::vector<int>(n);
            std::iota(p.begin()+1, p.end(), 0);
            matrix_type m(memory::on_host(p));

            EXPECT_EQ(m.size(), n);
            EXPECT_EQ(m.num_cells(), 1);

            memory::fill(m.d(),  2);
            memory::fill(m.l(), -1);
            memory::fill(m.u(), -1);
            memory::fill(m.rhs(),1);

            m.solve();

            auto x = m.rhs();
            auto err = math::square(std::fabs(2.*x[0] - x[1] - 1.));
            for(auto i : make_span(1,n-1)) {
                err += math::square(std::fabs(2.*x[i] - x[i-1] - x[i+1] - 1.));
            }
            err += math::square(std::fabs(2.*x[n-1] - x[n-2] - 1.));

            EXPECT_NEAR(0., std::sqrt(err), 1e-8);
        }
    }
}

