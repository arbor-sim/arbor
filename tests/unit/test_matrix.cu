#include <numeric>
#include <vector>

#include "gtest.h"

#include <math.hpp>
#include <matrix.hpp>
#include <util/span.hpp>

TEST(matrix, solve_gpu)
{
    using nest::mc::util::make_span;
    using matrix_type = nest::mc::matrix<double, int, nest::mc::backends::gpu::matrix_policy>;

    // trivial case : 1x1 matrix
    {
        matrix_type m(memory::on_host(std::vector<int>{0}));

        memory::fill(m.d(),  2);
        memory::fill(m.l(), -1);
        memory::fill(m.u(), -1);
        memory::fill(m.rhs(),1);

        m.solve();

        auto rhs = memory::on_host(m.rhs());

        EXPECT_EQ(rhs[0], 0.5);
    }

    // matrices in the range of 2x2 to 100x100
    {
        using namespace nest::mc;
        for(auto n : make_span(2u,101u)) {
            auto p = std::vector<int>(n);
            std::iota(p.begin()+1, p.end(), 0);
            matrix_type m{memory::on_host(p)};

            EXPECT_EQ(m.size(), n);
            EXPECT_EQ(m.num_cells(), 1);

            memory::fill(m.d(),  2);
            memory::fill(m.l(), -1);
            memory::fill(m.u(), -1);
            memory::fill(m.rhs(),1);

            m.solve();

            auto x = memory::on_host(m.rhs());
            auto err = math::square(std::fabs(2.*x[0] - x[1] - 1.));
            for(auto i : make_span(1,n-1)) {
                err += math::square(std::fabs(2.*x[i] - x[i-1] - x[i+1] - 1.));
            }
            err += math::square(std::fabs(2.*x[n-1] - x[n-2] - 1.));

            EXPECT_NEAR(0., std::sqrt(err), 1e-8);
        }
    }
}
