/*
 * Unit tests for em_morphology, region, locset, label_dict.
 */

#include <cmath>
#include <string>
#include <vector>

#include "../test/gtest.h"

#include <arbor/math.hpp>
#include <arbor/morph/error.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

#include "io/sepval.hpp"
#include "morph/em_morphology.hpp"
#include "util/span.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
    return o << "[" << arb::io::csv(v) << "]";
}

// Test the morphology meta-data that is cached on construction of
// em_morpholgy, e.g. interpolation information and terminal nodes.
TEST(em_morphology, cache) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    using loc = arb::mlocation;
    constexpr auto npos = arb::mnpos;

    // A single unbranched cable with 5 sample points.
    // The cable has length 10 μm, with samples located at
    // 0 μm, 1 μm, 3 μm, 7 μm and 10 μm.
    {
        pvec parents = {npos, 0, 1, 2, 3};
        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{  1,  0,  0,  2}, 1},
            {{  3,  0,  0,  2}, 1},
            {{  7,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 1},
        };
        arb::sample_tree sm(samples, parents);
        arb::em_morphology em(arb::morphology(sm, false));
        EXPECT_EQ(em.sample2loc(0), (loc{0,0}));
        EXPECT_EQ(em.sample2loc(1), (loc{0,0.1}));
        EXPECT_EQ(em.sample2loc(2), (loc{0,0.3}));
        EXPECT_EQ(em.sample2loc(3), (loc{0,0.7}));
        EXPECT_EQ(em.sample2loc(4), (loc{0,1}));

        EXPECT_EQ(em.root(),      (loc{0,0}));
        EXPECT_EQ(em.terminals(), (arb::mlocation_list{{0,1}}));
    }

    // Eight samples
    //
    //  sample ids:
    //            0
    //           1 3
    //          2   4
    //             5 6
    //                7
    {   // Spherical root.
        pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};

        svec samples = {
            {{  0,  0,  0, 10}, 1},
            {{ 10,  0,  0,  2}, 3},
            {{100,  0,  0,  2}, 3},
            {{  0, 10,  0,  2}, 3},
            {{  0,100,  0,  2}, 3},
            {{100,100,  0,  2}, 3},
            {{  0,200,  0,  2}, 3},
            {{  0,300,  0,  2}, 3},
        };
        arb::sample_tree sm(samples, parents);
        arb::em_morphology em(arb::morphology(sm, true));

        EXPECT_EQ(em.sample2loc(0), (loc{0,0}));
        EXPECT_EQ(em.sample2loc(1), (loc{1,0}));
        EXPECT_EQ(em.sample2loc(2), (loc{1,1}));
        EXPECT_EQ(em.sample2loc(3), (loc{2,0}));
        EXPECT_EQ(em.sample2loc(4), (loc{2,1}));
        EXPECT_EQ(em.sample2loc(5), (loc{3,1}));
        EXPECT_EQ(em.sample2loc(6), (loc{4,0.5}));
        EXPECT_EQ(em.sample2loc(7), (loc{4,1}));

        EXPECT_EQ(em.root(),      (loc{0,0}));
        EXPECT_EQ(em.terminals(), (arb::mlocation_list{{1,1}, {3,1}, {4,1}}));
    }
    {   // No Spherical root
        pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};

        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 3},
            {{100,  0,  0,  2}, 3},
            {{  0, 10,  0,  2}, 3},
            {{  0,100,  0,  2}, 3},
            {{100,100,  0,  2}, 3},
            {{  0,130,  0,  2}, 3},
            {{  0,300,  0,  2}, 3},
        };
        arb::sample_tree sm(samples, parents);
        arb::em_morphology em(arb::morphology(sm, false));

        EXPECT_EQ(em.sample2loc(0), (loc{0,0}));
        EXPECT_EQ(em.sample2loc(1), (loc{0,0.1}));
        EXPECT_EQ(em.sample2loc(2), (loc{0,1}));
        EXPECT_EQ(em.sample2loc(3), (loc{1,0.1}));
        EXPECT_EQ(em.sample2loc(4), (loc{1,1}));
        EXPECT_EQ(em.sample2loc(5), (loc{2,1}));
        EXPECT_EQ(em.sample2loc(6), (loc{3,0.15}));
        EXPECT_EQ(em.sample2loc(7), (loc{3,1}));

        EXPECT_EQ(em.root(),      (loc{0,0}));
        EXPECT_EQ(em.terminals(), (arb::mlocation_list{{0,1}, {2,1}, {3,1}}));
    }
}

TEST(em_morphology, cover) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    using ll = arb::mlocation_list;
    constexpr auto npos = arb::mnpos;
    using arb::util::make_span;

    // Eight samples
    //            0
    //           1 3
    //          2   4
    //             5 6
    //                7
    pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};

    svec samples = {
        {{  0,  0,  0, 10}, 1},
        {{ 10,  0,  0,  2}, 3},
        {{100,  0,  0,  2}, 3},
        {{  0, 10,  0,  2}, 3},
        {{  0,100,  0,  2}, 3},
        {{100,100,  0,  2}, 3},
        {{  0,200,  0,  2}, 3},
        {{  0,300,  0,  2}, 3},
    };
    arb::sample_tree sm(samples, parents);

    // non-spherical root.
    {
        arb::em_morphology em(arb::morphology(sm, false));

        for (arb::msize_t i: make_span(em.morph().num_branches())) {
            for (auto j: make_span(1,9)) {
                arb::mlocation l{i, j/10.};
                EXPECT_EQ(em.cover(l), (ll{l}));
            }
        }

        EXPECT_EQ(em.cover({0,0}), (ll{{0,0}, {1,0}}));
        EXPECT_EQ(em.cover({0,1}), (ll{{0,1}}));
        EXPECT_EQ(em.cover({1,1}), (ll{{1,1}, {2,0}, {3,0}}));
        EXPECT_EQ(em.cover({2,1}), (ll{{2,1}}));
        EXPECT_EQ(em.cover({3,1}), (ll{{3,1}}));

        EXPECT_EQ(em.cover({1,0}), (ll{{0,0}, {1,0}}));
        EXPECT_EQ(em.cover({2,0}), (ll{{1,1}, {2,0}, {3,0}}));
        EXPECT_EQ(em.cover({3,0}), (ll{{1,1}, {2,0}, {3,0}}));
    }
    // spherical root.
    {
        arb::em_morphology em(arb::morphology(sm, true));

        EXPECT_EQ(em.cover({0,0}), (ll{{0,0}}));
        EXPECT_EQ(em.cover({0,1}), (ll{{0,1}, {1,0}, {2,0}}));
        EXPECT_EQ(em.cover({1,1}), (ll{{1,1}}));
        EXPECT_EQ(em.cover({2,1}), (ll{{2,1}, {3,0}, {4,0}}));
        EXPECT_EQ(em.cover({3,1}), (ll{{3,1}}));
        EXPECT_EQ(em.cover({4,1}), (ll{{4,1}}));
        EXPECT_EQ(em.cover({1,0}), (ll{{0,1}, {1,0}, {2,0}}));
        EXPECT_EQ(em.cover({2,0}), (ll{{0,1}, {1,0}, {2,0}}));
        EXPECT_EQ(em.cover({3,0}), (ll{{2,1}, {3,0}, {4,0}}));
        EXPECT_EQ(em.cover({4,0}), (ll{{2,1}, {3,0}, {4,0}}));
    }
}

TEST(em_morphology, minset) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    using ll = arb::mlocation_list;
    constexpr auto npos = arb::mnpos;

    // Eight samples
    //                  no-sphere  sphere
    //          sample   branch    branch
    //            0         0         0
    //           1 3       0 1       1 2
    //          2   4     0   1     1   2
    //             5 6       2 3       3 4
    //                7         3         4
    pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};
    svec samples = {
        {{  0,  0,  0,  2}, 3},
        {{ 10,  0,  0,  2}, 3},
        {{100,  0,  0,  2}, 3},
        {{  0, 10,  0,  2}, 3},
        {{  0,100,  0,  2}, 3},
        {{100,100,  0,  2}, 3},
        {{  0,200,  0,  2}, 3},
        {{  0,300,  0,  2}, 3},
    };
    arb::sample_tree sm(samples, parents);
    {
        arb::em_morphology em(arb::morphology(sm, false));

        EXPECT_EQ((ll{}), em.minset(ll{}));
        EXPECT_EQ((ll{{2,0.1}}), em.minset(ll{{2,0.1}}));
        EXPECT_EQ((ll{{0,0.5}, {1,0.5}}), em.minset(ll{{0,0.5}, {1,0.5}}));
        EXPECT_EQ((ll{{0,0.5}}), em.minset(ll{{0,0.5}}));
        EXPECT_EQ((ll{{0,0}, {1,0}}), em.minset(ll{{0,0}, {0,0.5}, {1,0}, {1,0.5}}));
        EXPECT_EQ((ll{{0,0}, {1,0.5}}), em.minset(ll{{0,0}, {0,0.5}, {1,0.5}, {2,0.5}}));
        EXPECT_EQ((ll{{0,0}, {2,0.5}}), em.minset(ll{{0,0}, {0,0.5}, {2,0.5}}));
        EXPECT_EQ((ll{{0,0}, {2,0.5}, {3,0}}), em.minset(ll{{0,0}, {0,0.5}, {2,0.5}, {3,0}, {3,1}}));
        EXPECT_EQ((ll{{0,0}, {1,0}}), em.minset(ll{{0,0}, {0,0.5}, {1,0}, {2,0.5}, {3,0}, {3,1}}));
    }
    {
        arb::em_morphology em(arb::morphology(sm, true));

        EXPECT_EQ((ll{}), em.minset(ll{}));
        EXPECT_EQ((ll{{2,0.1}}), em.minset(ll{{2,0.1}}));
        EXPECT_EQ((ll{{0,0.5}}), em.minset(ll{{0,0.5}, {1,0.5}}));
        EXPECT_EQ((ll{{0,0.5}}), em.minset(ll{{0,0.5}}));
        EXPECT_EQ((ll{{0,0}}), em.minset(ll{{0,0}, {0,0.5}, {1,0}, {1,0.5}}));
        EXPECT_EQ((ll{{1,0.5}, {3,0.1}, {4,0.5}}), em.minset(ll{{1,0.5}, {1,1}, {3,0.1}, {4,0.5}, {4,0.7}}));
    }
}

// Test expressions that describe location sets (locset).
TEST(locset, expressions) {
    auto root = arb::ls::root();
    auto term = arb::ls::terminal();
    auto samp = arb::ls::sample(42);
    auto loc = arb::ls::location({2, 0.5});

    auto to_string = [](auto&& x) {
        std::stringstream s;
        s << x;
        return s.str();
    };

    EXPECT_EQ(to_string(root), "root");
    EXPECT_EQ(to_string(term), "terminal");
    EXPECT_EQ(to_string(sum(root, term)), "(sum root terminal)");
    EXPECT_EQ(to_string(sum(root, term, samp)),
            "(sum (sum root terminal) (sample 42))");
    EXPECT_EQ(to_string(sum(root, term, samp, loc)),
            "(sum (sum (sum root terminal) (sample 42)) (location 2 0.5))");
    EXPECT_EQ(to_string(samp), "(sample 42)");
    EXPECT_EQ(to_string(loc), "(location 2 0.5)");

    // Location positions have to be in the range [0,1].
    // Assert that an exception is thrown if and invalide location is requested.
    EXPECT_THROW(arb::ls::location({2, 1.5}), arb::morphology_error);
    EXPECT_THROW(arb::ls::location({arb::mnpos, 0.5}), arb::morphology_error);
}

// Test expressions that describe regions.
TEST(region, expressions) {
    using arb::reg::cable;
    auto to_string = [](auto&& x) {
        std::stringstream s;
        s << x;
        return s.str();
    };

    auto c1 = arb::reg::cable({1,0,1});
    auto c2 = arb::reg::cable({4,0.1,0.5});
    auto c3 = join(cable({4,0.1,0.5}), cable({3,0,1}));
    auto b1 = arb::reg::branch(1);
    auto t1 = arb::reg::tagged(1);
    auto t2 = arb::reg::tagged(2);
    auto t3 = arb::reg::tagged(3);
    auto all = arb::reg::all();

    EXPECT_EQ(to_string(c1), "(cable 1 0 1)");
    EXPECT_EQ(to_string(c2), "(cable 4 0.1 0.5)");
    EXPECT_EQ(to_string(c3), "(join (cable 4 0.1 0.5) (cable 3 0 1))");
    EXPECT_EQ(to_string(b1), "(cable 1 0 1)");
    EXPECT_EQ(to_string(t1), "(tag 1)");
    EXPECT_EQ(to_string(t2), "(tag 2)");
    EXPECT_EQ(to_string(intersect(c1, t2)), "(intersect (cable 1 0 1) (tag 2))");
    EXPECT_EQ(to_string(join(c1, t2)),  "(join (cable 1 0 1) (tag 2))");
    EXPECT_EQ(to_string(join(t1, t2, t3)), "(join (join (tag 1) (tag 2)) (tag 3))");
    EXPECT_EQ(to_string(intersect(t1, t2, t3)), "(intersect (intersect (tag 1) (tag 2)) (tag 3))");
    EXPECT_EQ(to_string(intersect(join(c1, t2), c2)),  "(intersect (join (cable 1 0 1) (tag 2)) (cable 4 0.1 0.5))");
    EXPECT_EQ(to_string(all), "all");

    EXPECT_THROW(arb::reg::cable({1, 0, 1.1}), arb::morphology_error);
    EXPECT_THROW(arb::reg::branch(-1), arb::morphology_error);
}

TEST(locset, thingify) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    using ll = arb::mlocation_list;
    const auto npos = arb::mnpos;

    auto root = arb::ls::root();
    auto term = arb::ls::terminal();
    auto samp = arb::ls::sample(4);
    auto midb2 = arb::ls::location({2, 0.5});
    auto midb1 = arb::ls::location({1, 0.5});
    auto begb0 = arb::ls::location({0, 0});
    auto begb1 = arb::ls::location({1, 0});
    auto begb2 = arb::ls::location({2, 0});
    auto begb3 = arb::ls::location({3, 0});
    auto begb4 = arb::ls::location({4, 0});

    // Eight samples
    //
    //            0
    //           1 3
    //          2   4
    //             5 6
    //                7
    pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};
    svec samples = {
        {{  0,  0,  0,  2}, 3},
        {{ 10,  0,  0,  2}, 3},
        {{100,  0,  0,  2}, 3},
        {{  0, 10,  0,  2}, 3},
        {{  0,100,  0,  2}, 3},
        {{100,100,  0,  2}, 3},
        {{  0,200,  0,  2}, 3},
        {{  0,300,  0,  2}, 3},
    };
    arb::sample_tree sm(samples, parents);

    {
        arb::em_morphology em(arb::morphology(sm, true));

        EXPECT_EQ(thingify(root, em),  (ll{{0,0}}));
        EXPECT_EQ(thingify(term, em),  (ll{{1,1},{3,1},{4,1}}));
        EXPECT_EQ(thingify(samp, em),  (ll{{2,1}}));
        EXPECT_EQ(thingify(midb2, em), (ll{{2,0.5}}));
        EXPECT_EQ(thingify(midb1, em), (ll{{1,0.5}}));
        EXPECT_EQ(thingify(begb0, em), (ll{{0,0}}));
        EXPECT_EQ(thingify(begb1, em), (ll{{0,1}}));
        EXPECT_EQ(thingify(begb2, em), (ll{{0,1}}));
        EXPECT_EQ(thingify(begb3, em), (ll{{2,1}}));
        EXPECT_EQ(thingify(begb4, em), (ll{{2,1}}));
    }
    {
        arb::em_morphology em(arb::morphology(sm, false));

        EXPECT_EQ(thingify(root, em),  (ll{{0,0}}));
        EXPECT_EQ(thingify(term, em),  (ll{{0,1},{2,1},{3,1}}));
        EXPECT_EQ(thingify(samp, em),  (ll{{1,1}}));
        EXPECT_EQ(thingify(midb2, em), (ll{{2,0.5}}));
        EXPECT_EQ(thingify(midb1, em), (ll{{1,0.5}}));
        EXPECT_EQ(thingify(begb0, em), (ll{{0,0}}));
        EXPECT_EQ(thingify(begb1, em), (ll{{0,0}}));
        EXPECT_EQ(thingify(begb2, em), (ll{{1,1}}));
        EXPECT_EQ(thingify(begb3, em), (ll{{1,1}}));
        // In the absence of a spherical root, there is no branch 4.
        EXPECT_THROW(thingify(begb4, em), arb::morphology_error);
    }
}

// Forward declare implementation of join, union & intersect on location lists
// for testing.
namespace arb { namespace ls {
    mlocation_list intersection(const mlocation_list& lhs, const mlocation_list& rhs);
    mlocation_list join(const mlocation_list& lhs, const mlocation_list& rhs);
    mlocation_list sum(const mlocation_list& lhs, const mlocation_list& rhs);
}}

arb::mlocation_list ml(std::vector<int> bids) {
    arb::mlocation_list L;
    for (arb::msize_t b: bids) L.push_back({b, 0});
    return L;
}


// Test out multiset joins, intersections and sums
TEST(locset, join_intersect_sum) {
    using ll = arb::mlocation_list;
    using arb::ls::intersection;
    using arb::ls::sum;
    using arb::ls::join;

    {
        ll lhs{};
        ll rhs{};
        EXPECT_EQ(sum(lhs, rhs), ll{});
        EXPECT_EQ(join(lhs, rhs), ll{});
        EXPECT_EQ(intersection(lhs, rhs), ll{});
    }
    {
        ll lhs{};
        ll rhs = ml({0,1});
        EXPECT_EQ(sum(lhs, rhs), rhs);
        EXPECT_EQ(join(lhs, rhs), rhs);
        EXPECT_EQ(intersection(lhs, rhs), ll{});
    }
    {
        ll lhs = ml({1});
        ll rhs = ml({1});
        EXPECT_EQ(sum(lhs,  rhs), ml({1,1}));
        EXPECT_EQ(join(lhs, rhs), ml({1}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1}));
    }
    {
        ll lhs = ml({1,1});
        ll rhs = ml({1});
        EXPECT_EQ(sum(lhs,  rhs), ml({1,1,1}));
        EXPECT_EQ(join(lhs, rhs), ml({1,1}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1}));
    }
    {
        ll lhs = ml({0,3});
        ll rhs = ml({1,2});
        EXPECT_EQ(sum(lhs,  rhs), ml({0,1,2,3}));
        EXPECT_EQ(join(lhs, rhs), ml({0,1,2,3}));
        EXPECT_EQ(intersection(lhs, rhs), ll{});
    }
    {
        ll lhs = ml({0,1,3});
        ll rhs = ml({0,1,3});
        EXPECT_EQ(sum(lhs, rhs), ml({0,0,1,1,3,3}));
        EXPECT_EQ(join(lhs, rhs), lhs);
        EXPECT_EQ(intersection(lhs, rhs), lhs);
    }
    {
        ll lhs = ml({0,1,3});
        ll rhs = ml({1,2});
        EXPECT_EQ(sum(lhs, rhs), ml({0,1,1,2,3}));
        EXPECT_EQ(join(lhs, rhs), ml({0,1,2,3}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1}));
    }
    {
        ll lhs = ml({0,1,1,3});
        ll rhs = ml({1,2});
        EXPECT_EQ(sum(lhs, rhs), ml({0,1,1,1,2,3}));
        EXPECT_EQ(join(lhs, rhs), ml({0,1,1,2,3}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1}));
    }
    {
        ll lhs = ml({0,1,1,3,5,5,12});
        ll rhs = ml({1,2,2,5,5,5});
        EXPECT_EQ(sum(lhs, rhs),  ml({0,1,1,1,2,2,3,5,5,5,5,5,12}));
        EXPECT_EQ(join(lhs, rhs), ml({0,1,1,2,2,3,5,5,5,12}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1,5,5}));
    }
}

TEST(region, thingify) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    //using cab = arb::mcable;
    using cl = arb::mcable_list;
    constexpr auto npos = arb::mnpos;

    // A single unbranched cable with 5 sample points.
    // The cable has length 10 μm, with samples located at
    // 0 μm, 1 μm, 3 μm, 7 μm and 10 μm.
    {
        pvec parents = {npos, 0, 1, 2, 3};
        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{  1,  0,  0,  2}, 1},
            {{  3,  0,  0,  2}, 2},
            {{  7,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 2},
        };
        arb::sample_tree sm(samples, parents);
        arb::em_morphology em(arb::morphology(sm, false));

        auto h1  = arb::reg::cable({0, 0, 0.5});
        auto h2  = arb::reg::cable({0, 0.5, 1});
        auto t1  = arb::reg::tagged(1);
        auto t2  = arb::reg::tagged(2);
        auto all = arb::reg::all();

        // Concrete cable lists
        cl h1_{{0, 0.0, 0.5}};
        cl h2_{{0, 0.5, 1.0}};
        cl t1_{{0, 0.0, 0.1}, {0, 0.3, 0.7}};
        cl t2_{{0, 0.1, 0.3}, {0, 0.7, 1.0}};
        cl all_{{0, 0, 1}};
        cl empty_{};

        EXPECT_EQ(thingify(h1, em), h1_);
        EXPECT_EQ(thingify(h2, em), h2_);
        EXPECT_EQ(thingify(t1, em), t1_);
        EXPECT_EQ(thingify(t2, em), t2_);
        EXPECT_EQ(thingify(join(h1, h2), em), all_);

        EXPECT_EQ(thingify(intersect(h1, h2), em), (cl{{0, 0.5, 0.5}}));

        EXPECT_EQ(thingify(intersect(h1, h1), em), h1_);
        EXPECT_EQ(thingify(intersect(t1, t1), em), t1_);
        EXPECT_EQ(thingify(join(t1, t2), em), all_);
        EXPECT_EQ(thingify(intersect(all, t1), em), t1_);
        EXPECT_EQ(thingify(intersect(all, t2), em), t2_);
        EXPECT_EQ(thingify(join(all, t1), em), all_);
        EXPECT_EQ(thingify(join(all, t2), em), all_);
        EXPECT_EQ(thingify(join(h1, t1), em), (cl{{0, 0, 0.7}}));
        EXPECT_EQ(thingify(join(h1, t2), em), (cl{{0, 0, 0.5}, {0, 0.7, 1}}));
        EXPECT_EQ(thingify(intersect(h2, t1), em), (cl{{0, 0.5, 0.7}}));
    }


    // Test handling of spherical soma on multi-branch morphologies
    //
    //  sample ids:
    //              0           |
    //            1   3         |
    //          2       4       |
    //  tags:
    //              1           |
    //            3   2         |
    //          3       2       |
    {
        pvec parents = {npos, 0, 1, 0, 3};
        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 3},
            {{100,  0,  0,  2}, 3},
            {{  0, 10,  0,  2}, 2},
            {{  0,100,  0,  2}, 2},
        };

        // with a spherical root
        arb::sample_tree sm(samples, parents);
        arb::em_morphology em(arb::morphology(sm, true));

        using arb::reg::tagged;
        using arb::reg::branch;
        using arb::reg::all;

        EXPECT_EQ(thingify(tagged(1), em), (arb::mcable_list{{0,0,1}}));
        EXPECT_EQ(thingify(tagged(2), em), (arb::mcable_list{{2,0,1}}));
        EXPECT_EQ(thingify(tagged(3), em), (arb::mcable_list{{1,0,1}}));
        EXPECT_EQ(thingify(join(tagged(1), tagged(2), tagged(3)), em), (arb::mcable_list{{0,0,1}, {1,0,1}, {2,0,1}}));
        EXPECT_EQ(thingify(join(tagged(1), tagged(2), tagged(3)), em), thingify(all(), em));
    }

    // Test multi-level morphologies.
    //
    // Eight samples
    //
    //  sample ids:
    //            0
    //           1 3
    //          2   4
    //             5 6
    //                7
    //  tags:
    //            1
    //           3 2
    //          3   2
    //             4 3
    //                3
    {
        pvec parents = {npos, 0, 1, 0, 3, 4, 4, 6};
        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 3},
            {{100,  0,  0,  2}, 3},
            {{  0, 10,  0,  2}, 2},
            {{  0,100,  0,  2}, 2},
            {{100,100,  0,  2}, 4},
            {{  0,200,  0,  2}, 3},
            {{  0,300,  0,  2}, 3},
        };
        arb::sample_tree sm(samples, parents);

        // Without spherical root
        arb::em_morphology em(arb::morphology(sm, false));

        using arb::reg::tagged;
        using arb::reg::branch;
        using arb::reg::all;
        using arb::reg::cable;
        using arb::mcable;
        auto soma = tagged(1);
        auto axon = tagged(2);
        auto dend = tagged(3);
        auto apic = tagged(4);
        auto b1  = branch(1);
        auto b3  = branch(3);
        auto b13 = join(b1, b3);

        cl empty_{};
        cl soma_ = empty_;

        // Whole branches:
        mcable b0_{0,0,1};
        mcable b1_{1,0,1};
        mcable b2_{2,0,1};
        mcable b3_{3,0,1};
        cl all_  = {b0_,b1_,b2_,b3_};

        mcable end1_{1,1,1};
        mcable root_{0,0,0};

        EXPECT_EQ(thingify(all(), em), all_);
        EXPECT_EQ(thingify(soma, em), empty_);
        EXPECT_EQ(thingify(axon, em), (cl{b1_}));
        EXPECT_EQ(thingify(dend, em), (cl{b0_,b3_}));
        EXPECT_EQ(thingify(apic, em), (cl{b2_}));
        EXPECT_EQ(thingify(join(dend, apic), em), (cl{b0_,b2_,b3_}));
        EXPECT_EQ(thingify(join(axon, join(dend, apic)), em), all_);

        // Test that intersection correctly generates zero-length cables at
        // parent-child interfaces.
        EXPECT_EQ(thingify(intersect(apic, dend), em), (cl{end1_}));
        EXPECT_EQ(thingify(intersect(apic, axon), em), (cl{end1_}));
        EXPECT_EQ(thingify(intersect(axon, dend), em), (cl{root_, end1_}));

        // Test some more interesting intersections and unions.

        //    123456789 123456789
        //   |---------|---------| lhs
        //   |  -----  |   ---   | rhs
        //   |  xxxxx  |   xxx   | rand
        //   |xxxxxxxxx|xxxxxxxxx| ror
        auto lhs  = b13;
        auto rhs  = join(cable({1,.2,.7}), cable({3,.3,.6}));
        auto rand = cl{         {1,.2,.7}, {3,.3,.6}};
        auto ror  = cl{         {1,.0,1.}, {3,.0,1.}};
        EXPECT_EQ(thingify(intersect(lhs, rhs), em), rand);
        EXPECT_EQ(thingify(join(lhs, rhs), em), ror);

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_EQ(thingify(intersect(lhs, rhs), em), rand);
        EXPECT_EQ(thingify(join(lhs, rhs), em), ror);

        //    123456789 123456789
        //   |   ----- | ----    | lhs
        //   |  -----  |   ---   | rhs
        //   |   xxxx  |   xx    | rand
        //   |  xxxxxx | xxxxx   | ror
        lhs  = join(cable({1,.3,.8}), cable({3,.1,.5}));
        rhs  = join(cable({1,.2,.7}), cable({3,.3,.6}));
        rand = cl{         {1,.3,.7}, {3,.3,.5}};
        ror  = cl{         {1,.2,.8}, {3,.1,.6}};
        EXPECT_EQ(thingify(intersect(lhs, rhs), em), rand);
        EXPECT_EQ(thingify(join(lhs, rhs), em), ror);

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_EQ(thingify(intersect(lhs, rhs), em), rand);
        EXPECT_EQ(thingify(join(lhs, rhs), em), ror);

        //    123456789 123456789
        //   | -- -    | --- --- | lhs
        //   |  -----  |   ---   | rhs
        //   |  x x    |   x x   | rand
        //   | xxxxxx  | xxxxxxx | ror
        lhs  = join(cable({1,.1,.3}), cable({1,.4,.5}), cable({3,.1,.4}), cable({3,.5,.9}));
        rhs  = join(cable({1,.2,.7}), cable({3,.3,.6}));
        rand = cl{         {1,.2,.3}, {1,.4,.5}, {3,.3,.4}, {3,.5,.6}};
        ror  = cl{         {1,.1,.7},            {3,.1,.9}};
        EXPECT_EQ(thingify(intersect(lhs, rhs), em), rand);
        EXPECT_EQ(thingify(join(lhs, rhs), em), ror);

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_EQ(thingify(intersect(lhs, rhs), em), rand);
        EXPECT_EQ(thingify(join(lhs, rhs), em), ror);
    }
}

