#include <numeric>
#include <vector>

#include "gtest.h"

#include "../src/math.hpp"
#include "../src/matrix.hpp"

TEST(matrix, construct_from_parent_only)
{
    using matrix_type = nest::mc::matrix<double, int>;

    // pass parent index as a std::vector by reference
    // the input should not be moved from
    {
        std::vector<int> p = {0,0,1};
        matrix_type m{p};
        EXPECT_EQ(m.num_cells(), 1);
        EXPECT_EQ(m.size(), 3u);
        EXPECT_EQ(p.size(), 3u);

        auto mp = m.p();
        EXPECT_EQ(mp[0], 0);
        EXPECT_EQ(mp[1], 0);
        EXPECT_EQ(mp[2], 1);
    }

    // pass parent index as a std::vector by rvalue reference
    // the input should not be moved from
    {
        std::vector<int> p = {0,0,1};
        matrix_type m{std::move(p)};
        EXPECT_EQ(m.num_cells(), 1);
        EXPECT_EQ(m.size(), 3u);
        EXPECT_EQ(p.size(), 3u);

        auto mp = m.p();
        EXPECT_EQ(mp[0], 0);
        EXPECT_EQ(mp[1], 0);
        EXPECT_EQ(mp[2], 1);
    }

    // pass parent index as a HostVector by reference
    // expect that the input is not moved from
    {
        memory::HostVector<int> p(3, 0);
        std::iota(p.begin()+1, p.end(), 0);
        matrix_type m{p};
        EXPECT_EQ(m.num_cells(), 1);
        EXPECT_EQ(m.size(), 3u);
        EXPECT_EQ(p.size(), 3u);

        auto mp = m.p();
        EXPECT_EQ(mp[0], 0);
        EXPECT_EQ(mp[1], 0);
        EXPECT_EQ(mp[2], 1);
    }
    // pass parent index as a HostVector by rvalue reference
    // expect that the input is moved from
    {
        memory::HostVector<int> p(3, 0);
        std::iota(p.begin()+1, p.end(), 0);
        matrix_type m{std::move(p)};
        EXPECT_EQ(m.num_cells(), 1);
        EXPECT_EQ(m.size(), 3u);
        EXPECT_EQ(p.size(), 0u); // 0 implies moved from

        auto mp = m.p();
        EXPECT_EQ(mp[0], 0);
        EXPECT_EQ(mp[1], 0);
        EXPECT_EQ(mp[2], 1);
    }
}

TEST(matrix, solve)
{
    using matrix_type = nest::mc::matrix<double, int>;

    // trivial case : 1x1 matrix
    {
        matrix_type m{std::vector<int>{0}};

        memory::fill(m.d(),  2);
        memory::fill(m.l(), -1);
        memory::fill(m.u(), -1);
        memory::fill(m.rhs(),1);

        m.solve();

        EXPECT_EQ(m.rhs()[0], 0.5);
    }
    // matrices in the range of 2x2 to 1000x1000
    {
        using namespace nest::mc::math;
        for(auto n : memory::Range(2,1001)) {
            auto p = std::vector<int>(n);
            std::iota(p.begin()+1, p.end(), 0);
            matrix_type m{p};

            EXPECT_EQ(m.size(), n);
            EXPECT_EQ(m.num_cells(), 1);

            memory::fill(m.d(),  2);
            memory::fill(m.l(), -1);
            memory::fill(m.u(), -1);
            memory::fill(m.rhs(),1);

            m.solve();

            auto x = m.rhs();
            auto err = square(std::fabs(2.*x[0] - x[1] - 1.));
            for(auto i : memory::Range(1,n-1)) {
                err += square(std::fabs(2.*x[i] - x[i-1] - x[i+1] - 1.));
            }
            err += square(std::fabs(2.*x[n-1] - x[n-2] - 1.));

            EXPECT_NEAR(0., std::sqrt(err), 1e-8);
        }
    }
}

