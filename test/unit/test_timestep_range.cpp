#include <cmath>

#include <gtest/gtest.h>

#include "timestep_range.hpp"

using namespace arb;

TEST(timestep_range, ctor) {
    {   // default ctor
        timestep_range r;
        EXPECT_TRUE(r.empty());
        EXPECT_EQ(r.size(), 0u);
        EXPECT_EQ(r.t_begin(), 0.);
        EXPECT_EQ(r.t_end(), 0.);
        EXPECT_EQ(r.begin(), r.end());
    }
    {   // from upper bound
        timestep_range r(10., 1.0);
        EXPECT_FALSE(r.empty());
        EXPECT_EQ(r.size(), 10u);
        EXPECT_EQ(r.t_begin(), 0.);
        EXPECT_EQ(r.t_end(), 10.);
    }
    {   // from invalid upper bound
        timestep_range r(-10., 1.0);
        EXPECT_TRUE(r.empty());
        EXPECT_EQ(r.size(), 0u);
        EXPECT_EQ(r.t_begin(), 0.);
        EXPECT_EQ(r.t_end(), -10.);
    }
    {   // from invalid dt
        timestep_range r(10., -1.0);
        EXPECT_FALSE(r.empty());
        EXPECT_EQ(r.size(), 1u);
        EXPECT_EQ(r.t_begin(), 0.);
        EXPECT_EQ(r.t_end(), 10.);
    }
    {   // from invalid upper bound and invalid dt
        timestep_range r(-10., -1.0);
        EXPECT_TRUE(r.empty());
        EXPECT_EQ(r.size(), 0u);
        EXPECT_EQ(r.t_begin(), 0.);
        EXPECT_EQ(r.t_end(), -10.);
    }
    {   // from lower and upper bound
        timestep_range r(5., 10., 1.0);
        EXPECT_FALSE(r.empty());
        EXPECT_EQ(r.size(), 5u);
        EXPECT_EQ(r.t_begin(), 5.);
        EXPECT_EQ(r.t_end(), 10.);
    }
    {   // from invalid upper bound
        timestep_range r(5., -10., 1.0);
        EXPECT_TRUE(r.empty());
        EXPECT_EQ(r.size(), 0u);
        EXPECT_EQ(r.t_begin(), 5.);
        EXPECT_EQ(r.t_end(), -10.);
    }
    {   // from invalid upper bound
        timestep_range r(15.0, 10., 1.0);
        EXPECT_TRUE(r.empty());
        EXPECT_EQ(r.size(), 0u);
        EXPECT_EQ(r.t_begin(), 15.);
        EXPECT_EQ(r.t_end(), 10.);
    }
    {   // from invalid dt
        timestep_range r(5., 10., -1.0);
        EXPECT_FALSE(r.empty());
        EXPECT_EQ(r.size(), 1u);
        EXPECT_EQ(r.t_begin(), 5.);
        EXPECT_EQ(r.t_end(), 10.);
    }
    {   // from invalid upper bound and invalid dt
        timestep_range r(5., -10., -1.0);
        EXPECT_TRUE(r.empty());
        EXPECT_EQ(r.size(), 0u);
        EXPECT_EQ(r.t_begin(), 5.);
        EXPECT_EQ(r.t_end(), -10.);
    }
    {   // from invalid upper bound and invalid dt
        timestep_range r(5., -10., -1.0);
        EXPECT_TRUE(r.empty());
        EXPECT_EQ(r.size(), 0u);
        EXPECT_EQ(r.t_begin(), 5.);
        EXPECT_EQ(r.t_end(), -10.);
    }
}

void check(const timestep_range& r, unsigned index, time_type t0, time_type t1) {
    EXPECT_EQ(r[index].t_begin(), t0);
    EXPECT_EQ(r[index].t_end(), t1);
    EXPECT_EQ((unsigned)(r.find(t0)-r.begin()), index);
    EXPECT_EQ((unsigned)(r.find(r[index].midpoint())-r.begin()), index);
    EXPECT_EQ((unsigned)(r.find(t1)-r.begin()), index+1);
}

TEST(timestep_range, range) {
    timestep_range r(5., 10., 1.0);
    check(r, 0, 5., 6.);
    check(r, 1, 6., 7.);
    check(r, 2, 7., 8.);
    check(r, 3, 8., 9.);
    check(r, 4, 9., 10.);

    EXPECT_EQ((r.begin()+1)->t_begin(), 6.);
    EXPECT_EQ((r.end()-4)->t_begin(), 6.);
    auto it_p = r.begin() += 1;
    auto it_m = r.begin() -= -1;
    EXPECT_EQ(it_p->t_begin(), 6.);
    EXPECT_EQ(it_m->t_begin(), 6.);
}

TEST(timestep_range, rounding) {
    auto t_end = 10.;
    for (std::size_t i=0; i<5; ++i) {
        t_end = std::nextafter(t_end, 11.0);
        timestep_range r(5., t_end, 1.0);
        EXPECT_FALSE(r.empty());
        EXPECT_EQ(r.size(), 5u);
        EXPECT_EQ(r.t_begin(), 5.);
        EXPECT_EQ(r.t_end(), t_end);
    }
}
