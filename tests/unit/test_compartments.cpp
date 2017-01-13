#include <cmath>
#include <string>

#include "../gtest.h"

#include <algorithms.hpp>
#include <compartment.hpp>

#include <math.hpp>
#include <util/span.hpp>
#include <util/transform.hpp>

using namespace nest::mc;
using namespace nest::mc::algorithms;
using namespace nest::mc::math;
using namespace nest::mc::util;

// not much to test here: just test that values passed into the constructor
// are correctly stored in members
TEST(compartments, compartment)
{
    {
        nest::mc::compartment c(100, 1.2, 2.1, 2.2);
        EXPECT_EQ(c.index, 100u);
        EXPECT_EQ(c.length, 1.2);
        EXPECT_EQ(c.radius.first, 2.1);
        EXPECT_EQ(c.radius.second, 2.2);

        auto c2 = c;
        EXPECT_EQ(c2.index, 100u);
        EXPECT_EQ(c2.length, 1.2);
        EXPECT_EQ(c2.radius.first, 2.1);
        EXPECT_EQ(c2.radius.second, 2.2);
    }

    {
        nest::mc::compartment c{100, 1, 2, 3};
        EXPECT_EQ(c.index, 100u);
        EXPECT_EQ(c.length, 1.);
        EXPECT_EQ(c.radius.first, 2.);
        EXPECT_EQ(c.radius.second, 3.);
    }
}

TEST(compartments, make_compartment_range)
{
    using namespace nest::mc;
    auto rng = make_compartment_range(10, 1.0, 2.0, 10.);

    EXPECT_EQ((*rng.begin()).index, 0u);
    EXPECT_EQ((*rng.end()).index, 10u);
    EXPECT_NE(rng.begin(), rng.end());

    unsigned count = 0;
    for (auto c : rng) {
        EXPECT_EQ(c.index, count);
        auto er = 1.0 + double(count)/10.;
        EXPECT_DOUBLE_EQ(c.radius.first, er);
        EXPECT_DOUBLE_EQ(c.radius.second, er+0.1);
        EXPECT_EQ(c.length, 1.0);
        ++count;
    }
    EXPECT_EQ(count, 10u);

    // test case of zero length range
    auto rng_empty = make_compartment_range(0, 1.0, 1.0, 0.);
    EXPECT_EQ(rng_empty.begin(), rng_empty.end());
}

// Divided compartments
// (FVM-friendly compartment data)

template <std::size_t N>
struct pw_cable_data {
    double radii[N+1];
    double lengths[N];

    static constexpr std::size_t nseg() { return N; }
    double r1() const { return radii[0]; }
    double r2() const { return radii[N]; }
    double length() const { return sum(lengths); }

    double area() const {
        return sum(transform_view(make_span(0, N),
            [&](unsigned i) { return area_frustrum(lengths[i], radii[i], radii[i+1]); }));
    }

    double volume() const {
        return sum(transform_view(make_span(0, N),
            [&](unsigned i) { return volume_frustrum(lengths[i], radii[i], radii[i+1]); }));
    }
};

pw_cable_data<1> cable_one = {
    {2.0, 5.0},
    {10.0}
};

pw_cable_data<4> cable_linear = {
    {2.0, 3.5, 6.0, 6.5, 6.75},
    {3.0, 5.0, 1.0, 0.5}
};

pw_cable_data<4> cable_jumble = {
    {2.0, 6.0, 3.5, 6.75, 6.5},
    {3.0, 5.0, 1.0, 0.5}
};

void expect_equal_divs(const div_compartment& da, const div_compartment& db) {
    EXPECT_EQ(da.index, db.index);

    double eps = std::numeric_limits<double>::epsilon();
    double e1 = std::min(da.length(), db.length())*8*eps;
    double e2 = std::min(da.area(), db.area())*8*eps;
    double e3 = std::min(da.volume(), db.volume())*8*eps;

    EXPECT_NEAR(da.left.length, db.left.length, e1);
    EXPECT_NEAR(da.left.area, db.left.area, e2);
    EXPECT_NEAR(da.left.volume, db.left.volume, e3);
    EXPECT_NEAR(da.left.radii.first, db.left.radii.first, e1);
    EXPECT_NEAR(da.left.radii.second, db.left.radii.second, e1);

    EXPECT_NEAR(da.right.length, db.right.length, e1);
    EXPECT_NEAR(da.right.area, db.right.area, e2);
    EXPECT_NEAR(da.right.volume, db.right.volume, e3);
    EXPECT_NEAR(da.right.radii.first, db.right.radii.first, e1);
    EXPECT_NEAR(da.right.radii.second, db.right.radii.second, e1);
}

TEST(compartments, div_ends) {
    using namespace math;

    {
        div_compartment_by_ends divcomps{1, cable_one.radii, cable_one.lengths};

        auto d = divcomps(0);
        auto r1 = cable_one.radii[0];
        auto r2 = cable_one.radii[1];
        auto l = cable_one.lengths[0];

        EXPECT_DOUBLE_EQ(r1, d.radii().first);
        EXPECT_DOUBLE_EQ(r2, d.radii().second);
        EXPECT_DOUBLE_EQ(l, d.length());
        EXPECT_DOUBLE_EQ(area_frustrum(l, r2, r1), d.area());
        EXPECT_DOUBLE_EQ(volume_frustrum(l, r2, r1), d.volume());

        auto sl = l/2.0;
        auto rc = mean(r1, r2);

        div_compartment expected{
            0,
            semi_compartment{sl, area_frustrum(sl, r1, rc), volume_frustrum(sl, r1, rc), {r1, rc}},
            semi_compartment{sl, area_frustrum(sl, rc, r2), volume_frustrum(sl, rc, r2), {rc, r2}}
        };

        SCOPED_TRACE("cable_one");
        expect_equal_divs(expected, d);
    }

    {
        // for a linear cable, expect this compartment maker to
        // create consistent compartments

        constexpr unsigned ncomp = 7;
        div_compartment_by_ends divlin{ncomp, cable_linear.radii, cable_linear.lengths};

        auto r1 = cable_linear.r1();
        auto r2 = cable_linear.r2();
        auto l = cable_linear.length();

        pw_cable_data<1> one = { {r1, r2}, {l} };
        div_compartment_by_ends divone{ncomp, one.radii, one.lengths};

        for (unsigned i=0; i<ncomp; ++i) {
            SCOPED_TRACE("cable_linear compartment "+std::to_string(i));
            auto da = divlin(i);
            auto db = divone(i);

            EXPECT_DOUBLE_EQ(l/ncomp, da.length());
            expect_equal_divs(da, db);
        }
    }
}

TEST(compartments, div_sample) {
    using namespace math;

    // expect by_ends and sampler to give same results on linear cable
    {
        constexpr unsigned ncomp = 7;
        div_compartment_sampler divsampler{ncomp, cable_linear.radii, cable_linear.lengths};
        div_compartment_by_ends divends{ncomp, cable_linear.radii, cable_linear.lengths};

        auto l = cable_linear.length();
        for (unsigned i=0; i<ncomp; ++i) {
            SCOPED_TRACE("cable_linear compartment "+std::to_string(i));
            auto da = divsampler(i);
            auto db = divends(i);

            EXPECT_DOUBLE_EQ(l/ncomp, da.length());
            expect_equal_divs(da, db);
        }
    }

    // expect (up to rounding) correct total area and volume if compartments
    // align with sub-segments; when they don't align, expect error to decrease
    // with ncomp
    {
        double area_expected = cable_jumble.area();
        double volume_expected = cable_jumble.volume();
        double eps = std::numeric_limits<double>::epsilon();

        double common_dx = 0.5; // depends on cable_jumble.lengths;
        // check our common_dx actually is ok
        for (double l: cable_jumble.lengths) {
            ASSERT_DOUBLE_EQ(l/common_dx, std::round(l/common_dx));
        }

        double length = cable_jumble.length();
        unsigned nbase = std::round(length/common_dx);

        for (unsigned m: {1u, 3u, 7u}) {
            unsigned ncomp = m*nbase;
            div_compartment_sampler divs{ncomp, cable_jumble.radii, cable_jumble.lengths};

            double area = sum(transform_view(make_span(0, ncomp),
                [&](unsigned i) { return divs(i).area(); }));

            double volume = sum(transform_view(make_span(0, ncomp),
                [&](unsigned i) { return divs(i).volume(); }));

            double e2 = std::min(area, area_expected)*ncomp*eps;
            double e3 = std::min(volume, volume_expected)*ncomp*eps;

            SCOPED_TRACE("cable_jumble ncomp "+std::to_string(ncomp));

            EXPECT_NEAR(area_expected, area, e2);
            EXPECT_NEAR(volume_expected, volume, e3);
        }

        double coarse_area = area_frustrum(length, cable_jumble.r1(), cable_jumble.r2());
        double coarse_volume = volume_frustrum(length, cable_jumble.r1(), cable_jumble.r2());
        double area_error = std::abs(area_expected-coarse_area);
        double volume_error = std::abs(volume_expected-coarse_volume);

        for (unsigned m: {1u, 10u, 100u}) {
            unsigned ncomp = m*nbase+1u;
            div_compartment_sampler divs{ncomp, cable_jumble.radii, cable_jumble.lengths};

            double area = sum(transform_view(make_span(0, ncomp),
                [&](unsigned i) { return divs(i).area(); }));

            double volume = sum(transform_view(make_span(0, ncomp),
                [&](unsigned i) { return divs(i).volume(); }));

            SCOPED_TRACE("cable_jumble ncomp "+std::to_string(ncomp));

            double err = std::abs(area_expected-area);
            EXPECT_LT(err, area_error);
            area_error = err;

            err = std::abs(volume_expected-volume);
            EXPECT_LT(err, volume_error);
            volume_error = err;
        }
    }
}

TEST(compartments, div_integrator) {
    using namespace math;

    // expect integrator and sampler to give same results on linear cable
    {
        constexpr unsigned ncomp = 7;
        div_compartment_sampler divintegrator{ncomp, cable_linear.radii, cable_linear.lengths};
        div_compartment_by_ends divends{ncomp, cable_linear.radii, cable_linear.lengths};

        auto l = cable_linear.length();
        for (unsigned i=0; i<ncomp; ++i) {
            SCOPED_TRACE("cable_linear compartment "+std::to_string(i));
            auto da = divintegrator(i);
            auto db = divends(i);

            EXPECT_DOUBLE_EQ(l/ncomp, da.length());
            expect_equal_divs(da, db);
        }
    }

    // expect integrator to give same (up to rounding) total areas and volumes
    // as the cable
    {
        double area_expected = cable_jumble.area();
        double volume_expected = cable_jumble.volume();
        double eps = std::numeric_limits<double>::epsilon();

        for (unsigned ncomp: make_span(1u, 23u)) {
            div_compartment_integrator divs{ncomp, cable_jumble.radii, cable_jumble.lengths};

            double area = sum(transform_view(make_span(0, ncomp),
                [&](unsigned i) { return divs(i).area(); }));

            double volume = sum(transform_view(make_span(0, ncomp),
                [&](unsigned i) { return divs(i).volume(); }));

            double e2 = std::min(area, area_expected)*ncomp*eps;
            double e3 = std::min(volume, volume_expected)*ncomp*eps;

            SCOPED_TRACE("cable_jumble ncomp "+std::to_string(ncomp));

            EXPECT_NEAR(area_expected, area, e2);
            EXPECT_NEAR(volume_expected, volume, e3);
        }
    }
}

