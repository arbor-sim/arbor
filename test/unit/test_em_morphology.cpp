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

#include "io/sepval.hpp"
#include "morph/morphology_impl.hpp"
#include "morph/em_morphology.hpp"
#include "util/span.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
    return o << "[" << arb::io::csv(v) << "]";
}

TEST(em_morphology, cache) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    using loc = arb::mlocation;
    constexpr auto npos = arb::mnpos;

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
    //              0           |
    //             / \          |
    //            1   3         |
    //           /     \        |
    //          2       4       |
    //                 / \      |
    //                5   6     |
    //                     \    |
    //                      7   |
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

TEST(region, expressions) {
}

TEST(locset, expressions) {
    using arb::ls::lor;
    using arb::ls::land;
    auto root = arb::ls::root();
    auto term = arb::ls::terminal();
    auto n1 = arb::ls::named("1");
    auto n2 = arb::ls::named("2");
    auto samp = arb::ls::sample(42);
    auto loc = arb::ls::location({2, 0.5});

    auto to_string = [](auto&& x) {
        std::stringstream s;
        s << x;
        return s.str();
    };

    EXPECT_EQ(to_string(root), "root");
    EXPECT_EQ(to_string(term), "terminal");
    EXPECT_EQ(to_string(n1), "\"1\"");
    EXPECT_EQ(to_string(n2), "\"2\"");
    EXPECT_EQ(to_string(land(root, term)), "(and root terminal)");
    EXPECT_EQ(to_string(lor(root, term)),  "(or root terminal)");
    EXPECT_EQ(to_string(lor(root, land(term, n1))),  "(or root (and terminal \"1\"))");
    EXPECT_EQ(to_string(samp), "(sample 42)");
    EXPECT_EQ(to_string(loc), "(location 2 0.5)");

    // Location positions have to be in the range [0,1].
    // Assert that an exception is thrown if and invalide location is requested.
    EXPECT_THROW(arb::ls::location({2, 1.5}), arb::morphology_error);
    EXPECT_THROW(arb::ls::location({arb::mnpos, 0.5}), arb::morphology_error);
}

TEST(locset, concretise) {
    using pvec = std::vector<arb::msize_t>;
    using svec = std::vector<arb::msample>;
    using ll = arb::mlocation_list;
    const auto npos = arb::mnpos;

    using arb::ls::lor;
    using arb::ls::land;
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
    //              0           |
    //             / \          |
    //            1   3         |
    //           /     \        |
    //          2       4       |
    //                 / \      |
    //                5   6     |
    //                     \    |
    //                      7   |
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

        EXPECT_EQ(concretise(root, em),  (ll{{0,0}}));
        EXPECT_EQ(concretise(term, em),  (ll{{1,1},{3,1},{4,1}}));
        EXPECT_EQ(concretise(samp, em),  (ll{{2,1}}));
        EXPECT_EQ(concretise(midb2, em), (ll{{2,0.5}}));
        EXPECT_EQ(concretise(midb1, em), (ll{{1,0.5}}));
        EXPECT_EQ(concretise(begb0, em), (ll{{0,0}}));
        EXPECT_EQ(concretise(begb1, em), (ll{{0,1}}));
        EXPECT_EQ(concretise(begb2, em), (ll{{0,1}}));
        EXPECT_EQ(concretise(begb3, em), (ll{{2,1}}));
        EXPECT_EQ(concretise(begb4, em), (ll{{2,1}}));
    }
    {
        arb::em_morphology em(arb::morphology(sm, false));

        EXPECT_EQ(concretise(root, em),  (ll{{0,0}}));
        EXPECT_EQ(concretise(term, em),  (ll{{0,1},{2,1},{3,1}}));
        EXPECT_EQ(concretise(samp, em),  (ll{{1,1}}));
        EXPECT_EQ(concretise(midb2, em), (ll{{2,0.5}}));
        EXPECT_EQ(concretise(midb1, em), (ll{{1,0.5}}));
        EXPECT_EQ(concretise(begb0, em), (ll{{0,0}}));
        EXPECT_EQ(concretise(begb1, em), (ll{{0,0}}));
        EXPECT_EQ(concretise(begb2, em), (ll{{1,1}}));
        EXPECT_EQ(concretise(begb3, em), (ll{{1,1}}));
        // In the absence of a spherical root, there is no branch 4.
        EXPECT_THROW(concretise(begb4, em), arb::morphology_error);
    }
}

TEST(region, concretise) {
}
