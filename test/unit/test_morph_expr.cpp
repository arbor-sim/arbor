#include "../test/gtest.h"

#include <vector>

#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/morph/sample_tree.hpp>

#include "util/span.hpp"
#include "util/strprintf.hpp"

#include "morph_pred.hpp"

using namespace arb;
using embedding = embed_pwlin;

using testing::region_eq;
using testing::cablelist_eq;
using testing::mlocationlist_eq;

TEST(region, expr_repn) {
    using util::to_string;

    auto c1 = reg::cable(1, 0, 1);
    auto c2 = reg::cable(4, 0.125, 0.5);
    auto c3 = join(reg::cable(4, 0.125, 0.5), reg::cable(3, 0, 1));
    auto b1 = reg::branch(1);
    auto t1 = reg::tagged(1);
    auto t2 = reg::tagged(2);
    auto t3 = reg::tagged(3);
    auto all = reg::all();

    EXPECT_EQ(to_string(c1), "(cable 1 0 1)");
    EXPECT_EQ(to_string(c2), "(cable 4 0.125 0.5)");
    EXPECT_EQ(to_string(c3), "(join (cable 4 0.125 0.5) (cable 3 0 1))");
    EXPECT_EQ(to_string(b1), "(cable 1 0 1)");
    EXPECT_EQ(to_string(t1), "(tag 1)");
    EXPECT_EQ(to_string(t2), "(tag 2)");
    EXPECT_EQ(to_string(intersect(c1, t2)), "(intersect (cable 1 0 1) (tag 2))");
    EXPECT_EQ(to_string(join(c1, t2)),  "(join (cable 1 0 1) (tag 2))");
    EXPECT_EQ(to_string(join(t1, t2, t3)), "(join (join (tag 1) (tag 2)) (tag 3))");
    EXPECT_EQ(to_string(intersect(t1, t2, t3)), "(intersect (intersect (tag 1) (tag 2)) (tag 3))");
    EXPECT_EQ(to_string(intersect(join(c1, t2), c2)),  "(intersect (join (cable 1 0 1) (tag 2)) (cable 4 0.125 0.5))");
    EXPECT_EQ(to_string(all), "(all)");
}

TEST(region, invalid_mcable) {
    EXPECT_NO_THROW(reg::cable(123, 0.5, 0.8));
    EXPECT_THROW(reg::cable(1, 0, 1.1), invalid_mcable);
    EXPECT_THROW(reg::branch(-1), invalid_mcable);
}

TEST(locset, expr_repn) {
    using util::to_string;

    auto root = ls::root();
    auto term = ls::terminal();
    auto samp = ls::sample(42);
    auto loc = ls::location(2, 0.5);

    EXPECT_EQ(to_string(root), "(root)");
    EXPECT_EQ(to_string(term), "(terminal)");
    EXPECT_EQ(to_string(sum(root, term)), "(sum (root) (terminal))");
    EXPECT_EQ(to_string(sum(root, term, samp)), "(sum (sum (root) (terminal)) (sample 42))");
    EXPECT_EQ(to_string(sum(root, term, samp, loc)), "(sum (sum (sum (root) (terminal)) (sample 42)) (location 2 0.5))");
    EXPECT_EQ(to_string(samp), "(sample 42)");
    EXPECT_EQ(to_string(loc), "(location 2 0.5)");
}

TEST(locset, invalid_mlocation) {
    // Location positions have to be in the range [0,1].
    EXPECT_NO_THROW(ls::location(123, 0.0));
    EXPECT_NO_THROW(ls::location(123, 0.02));
    EXPECT_NO_THROW(ls::location(123, 1.0));

    EXPECT_THROW(ls::location(0, 1.5), invalid_mlocation);
    EXPECT_THROW(ls::location(unsigned(-1), 0.), invalid_mlocation);
}

// Name evaluation (thingify) tests:

TEST(locset, thingify_named) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;

    locset banana = ls::root();
    locset cake = ls::terminal();

    sample_tree sm(svec{ {{0,0,0,1},1}, {{10,0,0,1},1} }, pvec{mnpos, 0});
    {
        label_dict dict;
        dict.set("banana", banana);
        dict.set("cake", cake);

        mprovider mp(morphology(sm, false), dict);
        EXPECT_EQ(thingify(locset("cake"), mp), thingify(cake, mp));
        EXPECT_EQ(thingify(locset("banana"), mp), thingify(banana, mp));

        EXPECT_THROW(thingify(locset("durian"), mp), unbound_name);
    }
    {
        label_dict dict;
        dict.set("banana", banana);
        dict.set("cake", cake);
        dict.set("topping", locset("fruit"));
        dict.set("fruit", locset("strawberry"));

        EXPECT_THROW(mprovider(morphology(sm, false), dict), unbound_name);
    }
    {
        label_dict dict;
        dict.set("banana", banana);
        dict.set("cake", cake);
        dict.set("topping", locset("fruit"));
        dict.set("fruit", sum(locset("banana"), locset("topping")));

        EXPECT_THROW(mprovider(morphology(sm, false), dict), circular_definition);
    }
}

TEST(region, thingify_named) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;

    region banana = reg::branch(0);
    region cake = reg::cable(0, 0.2, 0.3);

    // copy-paste ftw

    sample_tree sm(svec{ {{0,0,0,1},1}, {{10,0,0,1},1} }, pvec{mnpos, 0});
    {
        label_dict dict;
        dict.set("banana", banana);
        dict.set("cake", cake);

        mprovider mp(morphology(sm, false), dict);
        EXPECT_EQ(thingify(region("cake"), mp), thingify(cake, mp));
        EXPECT_EQ(thingify(region("banana"), mp), thingify(banana, mp));

        EXPECT_THROW(thingify(region("durian"), mp), unbound_name);
    }
    {
        label_dict dict;
        dict.set("banana", banana);
        dict.set("cake", cake);
        dict.set("topping", region("fruit"));
        dict.set("fruit", region("strawberry"));

        EXPECT_THROW(mprovider(morphology(sm, false), dict), unbound_name);
    }
    {
        label_dict dict;
        dict.set("banana", banana);
        dict.set("cake", cake);
        dict.set("topping", region("fruit"));
        dict.set("fruit", join(region("cake"), region("topping")));

        EXPECT_THROW(mprovider(morphology(sm, false), dict), circular_definition);
    }
}

// Embedded evaluation (thingify) tests:

TEST(locset, thingify) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    using ll = mlocation_list;
    auto root = ls::root();
    auto term = ls::terminal();
    auto samp = ls::sample(4);
    auto midb2 = ls::location(2, 0.5);
    auto midb1 = ls::location(1, 0.5);
    auto begb0 = ls::location(0, 0);
    auto begb1 = ls::location(1, 0);
    auto begb2 = ls::location(2, 0);
    auto begb3 = ls::location(3, 0);
    auto begb4 = ls::location(4, 0);
    auto multi = sum(begb3, midb2, midb1, midb2);

    // Eight samples
    //
    //            0
    //           1 3
    //          2   4
    //             5 6
    //                7
    pvec parents = {mnpos, 0, 1, 0, 3, 4, 4, 6};
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
    sample_tree sm(samples, parents);

    {
        mprovider mp(morphology(sm, true));

        EXPECT_EQ(thingify(root, mp),  (ll{{0,0}}));
        EXPECT_EQ(thingify(term, mp),  (ll{{1,1},{3,1},{4,1}}));
        EXPECT_EQ(thingify(samp, mp),  (ll{{2,1}}));
        EXPECT_EQ(thingify(midb2, mp), (ll{{2,0.5}}));
        EXPECT_EQ(thingify(midb1, mp), (ll{{1,0.5}}));
        EXPECT_EQ(thingify(begb0, mp), (ll{{0,0}}));
        EXPECT_EQ(thingify(begb1, mp), (ll{{1,0}}));
        EXPECT_EQ(thingify(begb2, mp), (ll{{2,0}}));
        EXPECT_EQ(thingify(begb3, mp), (ll{{3,0}}));
        EXPECT_EQ(thingify(begb4, mp), (ll{{4,0}}));

        // Check round-trip of implicit locset conversions.
        // (Use a locset which is non-trivially a multiset in order to
        // test the fold in the constructor.)
        EXPECT_EQ(thingify(multi, mp), thingify(locset(thingify(multi, mp)), mp));
    }
    {
        mprovider mp(morphology(sm, false));

        EXPECT_EQ(thingify(root, mp),  (ll{{0,0}}));
        EXPECT_EQ(thingify(term, mp),  (ll{{0,1},{2,1},{3,1}}));
        EXPECT_EQ(thingify(samp, mp),  (ll{{1,1}}));
        EXPECT_EQ(thingify(midb2, mp), (ll{{2,0.5}}));
        EXPECT_EQ(thingify(midb1, mp), (ll{{1,0.5}}));
        EXPECT_EQ(thingify(begb0, mp), (ll{{0,0}}));
        EXPECT_EQ(thingify(begb1, mp), (ll{{1,0}}));
        EXPECT_EQ(thingify(begb2, mp), (ll{{2,0}}));
        EXPECT_EQ(thingify(begb3, mp), (ll{{3,0}}));

        // In the absence of a spherical root, there is no branch 4.
        EXPECT_THROW(thingify(begb4, mp), no_such_branch);
    }
    {
        mprovider mp(morphology(sm, false));

        auto all = reg::all();
        auto ls0 = thingify(ls::uniform(all,  0,  9, 12), mp);
        auto ls1 = thingify(ls::uniform(all,  0,  9, 12), mp);
        auto ls2 = thingify(ls::uniform(all, 10, 19, 12), mp);
        auto ls3 = thingify(ls::uniform(all,  0,  9, 13), mp);
        auto ls4 = thingify(ls::uniform(all,  5,  6, 12), mp);
        auto ls5 = thingify(ls::uniform(all,  2,  5, 12), mp);
        auto ls6 = thingify(ls::uniform(all,  5, 11, 12), mp);

        EXPECT_EQ(ls0, ls1);

        bool found_none = true;
        for (auto l: ls2) {
            auto it = std::find(ls0.begin(), ls0.end(), l);
            if (it != ls0.end()) {
                found_none = false;
            }
        }
        EXPECT_TRUE(found_none);

        found_none = true;
        for (auto l: ls3) {
            auto it = std::find(ls0.begin(), ls0.end(), l);
            if (it != ls0.end()) {
                found_none = false;
            }
        }
        EXPECT_TRUE(found_none);

        bool found_all = true;
        for (auto l: ls4) {
            auto it = std::find(ls0.begin(), ls0.end(), l);
            if (it == ls0.end()) {
                found_all = false;
            }
        }
        EXPECT_TRUE(found_all);

        int found = 0;
        for (auto l: ls5) {
            auto it = std::find(ls4.begin(), ls4.end(), l);
            if (it != ls4.end()) found++;
        }
        EXPECT_TRUE(found == 1);

        found = 0;
        for (auto l: ls6) {
            auto it = std::find(ls4.begin(), ls4.end(), l);
            if (it != ls4.end()) found++;
        }
        EXPECT_TRUE(found == 2);
    }
    {
        mprovider mp(morphology(sm, false));
        auto sub_reg = join(reg::cable(0, 0.2, 0.7), reg::cable(1, 0.1, 1), reg::cable(3, 0.5, 0.6));

        auto ls0 = thingify(ls::uniform(sub_reg, 0, 10000, 72), mp);
        for (auto l: ls0) {
            switch(l.branch) {
                case 0: {
                    if (l.pos < 0.2 || l.pos > 0.7) FAIL();
                    break;
                }
                case 1: {
                    if (l.pos < 0.1 || l.pos > 1) FAIL();
                    break;
                }
                case 3: {
                    if (l.pos < 0.5 || l.pos > 0.6) FAIL();
                    break;
                }
                default: {
                    FAIL();
                    break;
                }
            }
            SUCCEED();
        }
    }
}

TEST(region, thingify_simple_morphologies) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    using cl = mcable_list;

    // A single unbranched cable with 5 sample points.
    // The cable has length 10 μm, with samples located at
    // 0 μm, 1 μm, 3 μm, 7 μm and 10 μm.
    {
        pvec parents = {mnpos, 0, 1, 2, 3};
        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{  1,  0,  0,  2}, 1},
            {{  3,  0,  0,  2}, 2},
            {{  7,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 2},
        };
        sample_tree sm(samples, parents);
        mprovider mp(morphology(sm, false));

        auto h1  = reg::cable(0, 0, 0.5);
        auto h2  = reg::cable(0, 0.5, 1);
        auto t1  = reg::tagged(1);
        auto t2  = reg::tagged(2);
        auto all = reg::all();

        // Concrete cable lists
        cl h1_{{0, 0.0, 0.5}};
        cl h2_{{0, 0.5, 1.0}};
        cl t1_{{0, 0.0, 0.1}, {0, 0.3, 0.7}};
        cl t2_{{0, 0.1, 0.3}, {0, 0.7, 1.0}};
        cl all_{{0, 0, 1}};
        cl empty_{};

        EXPECT_TRUE(region_eq(mp, join(h1, h2), all_));
        EXPECT_TRUE(region_eq(mp, intersect(h1, h2), cl{{0, 0.5, 0.5}}));
        EXPECT_TRUE(region_eq(mp, t1, t1_));
        EXPECT_TRUE(region_eq(mp, t2, t2_));
        EXPECT_TRUE(region_eq(mp, intersect(h1, h1), h1_));
        EXPECT_TRUE(region_eq(mp, intersect(t1, t1), t1_));
        EXPECT_TRUE(region_eq(mp, join(t1, t2), all_));
        EXPECT_TRUE(region_eq(mp, intersect(all, t1), t1_));
        EXPECT_TRUE(region_eq(mp, intersect(all, t2), t2_));
        EXPECT_TRUE(region_eq(mp, join(all, t1), all_));
        EXPECT_TRUE(region_eq(mp, join(all, t2), all_));
        EXPECT_TRUE(region_eq(mp, join(h1, t1), cl{{0, 0, 0.7}}));
        EXPECT_TRUE(region_eq(mp, join(h1, t2), cl{{0, 0, 0.5}, {0, 0.7, 1}}));
        EXPECT_TRUE(region_eq(mp, intersect(h2, t1), cl{{0, 0.5, 0.7}}));

        // Check round-trip of implicit region conversions.
        // (No fork points in cables, so extent should not including anyhing extra).
        EXPECT_EQ((mcable_list{{0, 0.3, 0.6}}), thingify(region(mcable{0, 0.3, 0.6}), mp).cables());
        EXPECT_TRUE(cablelist_eq(t2_, thingify(region(t2_), mp).cables()));
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
        pvec parents = {mnpos, 0, 1, 0, 3};
        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 3},
            {{100,  0,  0,  2}, 3},
            {{  0, 10,  0,  2}, 2},
            {{  0,100,  0,  2}, 2},
        };

        // with a spherical root
        sample_tree sm(samples, parents);
        mprovider mp(morphology(sm, true));

        using ls::location;
        using reg::tagged;
        using reg::distal_interval;
        using reg::proximal_interval;
        using reg::branch;
        using reg::cable;
        using reg::all;

        locset mid0_   = location(0,0.5);
        locset start1_ = location(1,0);
        locset end1_   = location(1,1);

        auto reg0_ = distal_interval(start1_, 45);
        auto reg1_ = distal_interval(mid0_,   74);
        auto reg2_ = proximal_interval(end1_, 45);
        auto reg3_ = proximal_interval(end1_, 91);
        auto reg4_ = distal_interval(end1_, 0);
        auto reg5_ = distal_interval(start1_, 0);
        auto reg6_ = proximal_interval(start1_, 0);

        EXPECT_TRUE(region_eq(mp, tagged(1), mcable_list{{0,0,1}}));
        EXPECT_TRUE(region_eq(mp, tagged(2), mcable_list{{2,0,1}}));
        EXPECT_TRUE(region_eq(mp, tagged(3), mcable_list{{1,0,1}}));
        EXPECT_TRUE(region_eq(mp, join(tagged(1), tagged(2), tagged(3)), mcable_list{{0,0,1}, {1,0,1}, {2,0,1}}));
        EXPECT_TRUE(region_eq(mp, join(tagged(1), tagged(2), tagged(3)), all()));
        EXPECT_TRUE(region_eq(mp, reg0_, mcable_list{{1,0,0.5}}));
        EXPECT_TRUE(region_eq(mp, reg1_, mcable_list{{0,0.5,1}, {1,0,0.8}, {2,0,0.8}}));
        EXPECT_TRUE(region_eq(mp, reg2_, mcable_list{{1,0.5,1}}));
        EXPECT_TRUE(region_eq(mp, reg3_, mcable_list{{0, 0.75, 1}, {1,0,1}}));
        EXPECT_TRUE(region_eq(mp, reg4_, mcable_list{{1,1,1}}));
        EXPECT_TRUE(region_eq(mp, reg5_, mcable_list{{0,1,1}}));
        EXPECT_TRUE(region_eq(mp, reg6_, mcable_list{{0,1,1}}));
    }
}

TEST(region, thingify_moderate_morphologies) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    using cl = mcable_list;

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
        pvec parents = {mnpos, 0, 1, 0, 3, 4, 4, 6};
        svec samples = {
            {{  0,  0,  0,  1}, 1},
            {{ 10,  0,  0,  1}, 3},
            {{100,  0,  0,  3}, 3},
            {{  0, 10,  0,  1}, 2},
            {{  0,100,  0,  5}, 2},
            {{100,100,  0,  2}, 4},
            {{  0,200,  0,  1}, 3},
            {{  0,300,  0,  3}, 3},
        };
        sample_tree sm(samples, parents);

        // Without spherical root
        mprovider mp(morphology(sm, false));

        using ls::location;
        using reg::tagged;
        using reg::distal_interval;
        using reg::proximal_interval;
        using reg::radius_lt;
        using reg::radius_le;
        using reg::radius_gt;
        using reg::radius_ge;
        using reg::branch;
        using reg::all;
        using reg::cable;

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

        mcable c_end1_{1,1,1};
        mcable c_root_{0,0,0};

        EXPECT_TRUE(region_eq(mp, all(), all_));
        EXPECT_TRUE(region_eq(mp, axon, cl{b1_}));
        EXPECT_TRUE(region_eq(mp, dend, cl{b0_,b3_}));
        EXPECT_TRUE(region_eq(mp, apic, cl{b2_}));
        EXPECT_TRUE(region_eq(mp, join(dend, apic), cl{b0_,b2_,b3_}));
        EXPECT_TRUE(region_eq(mp, join(axon, join(dend, apic)), all_));

        // Test that intersection correctly generates zero-length cables at
        // parent-child interfaces.
        EXPECT_TRUE(region_eq(mp, intersect(apic, dend), cl{c_end1_}));
        EXPECT_TRUE(region_eq(mp, intersect(apic, axon), cl{c_end1_}));
        EXPECT_TRUE(region_eq(mp, intersect(axon, dend), cl{c_root_, c_end1_}));

        // Test distal and proximal interavls
        auto start0_         = location(0, 0   );
        auto quar_1_         = location(1, 0.25);
        auto mid1_           = location(1, 0.5 );
        auto end1_           = location(1, 1   );
        auto mid2_           = location(2, 0.5 );
        auto end2_           = location(2, 1   );
        auto mid3_           = location(3, 0.5 );
        auto loc_3_0_        = location(3, 0.4 );
        auto loc_3_1_        = location(3, 0.65);
        auto mid_3_          = location(3, 0.5 );
        auto reg_a_ = join(cable(0,0.1,0.4), cable(2,0,1), cable(3,0.1,0.4));
        auto reg_b_ = join(cable(0,0.1,0.4), cable(2,0,1), cable(3,0.1,0.3));
        auto reg_c_ = join(cable(0,0,0.7), cable(2,0,0.5), cable(3,0.1,0.4), cable(3,0.9,1));
        auto reg_d_ = join(cable(0,0,0.7), cable(2,0,0.5), cable(3,0.1,0.9));

        // Distal from point and/or interval
        EXPECT_TRUE(region_eq(mp, distal_interval(start0_, 1000), mcable_list{{0,0,1}}));
        EXPECT_TRUE(region_eq(mp, distal_interval(quar_1_,  150), mcable_list{{1,0.25,1}, {2,0,0.75}, {3,0,0.375}}));
        EXPECT_TRUE(region_eq(mp, distal_interval(mid1_,   1000), mcable_list{{1,0.5,1}, {2,0,1}, {3,0,1}}));
        EXPECT_TRUE(region_eq(mp, distal_interval(mid1_,    150), mcable_list{{1,0.5,1}, {2,0,1}, {3,0,0.5}}));
        EXPECT_TRUE(region_eq(mp, distal_interval(end1_,    100), mcable_list{{2,0,1},{3,0,0.5}}));
        EXPECT_TRUE(region_eq(mp, distal_interval(join(quar_1_, mid1_),    150), mcable_list{{1,0.25,1}, {2,0,1}, {3,0,0.5}}));
        EXPECT_TRUE(region_eq(mp, distal_interval(join(quar_1_, loc_3_1_), 150), mcable_list{{1,0.25,1}, {2,0,0.75}, {3,0,0.375}, {3,0.65,1}}));
        EXPECT_TRUE(region_eq(mp, distal_interval(join(quar_1_, loc_3_1_), 150), mcable_list{{1,0.25,1}, {2,0,0.75}, {3,0,0.375}, {3,0.65,1}}));

        // Proximal from point and/or interval
        EXPECT_TRUE(region_eq(mp, proximal_interval(mid3_, 100), mcable_list{{3,0,0.5}}));
        EXPECT_TRUE(region_eq(mp, proximal_interval(mid3_, 150), mcable_list{{1,0.5,1}, {3,0,0.5}}));
        EXPECT_TRUE(region_eq(mp, proximal_interval(end2_, 150), mcable_list{{1,0.5,1}, {2,0,1}}));
        EXPECT_TRUE(region_eq(mp, proximal_interval(end2_, 500), mcable_list{{1,0,1}, {2,0,1}}));
        EXPECT_TRUE(region_eq(mp, proximal_interval(loc_3_0_, 100), mcable_list{{1,0.8,1}, {3,0,0.4}}));
        EXPECT_TRUE(region_eq(mp, proximal_interval(join(loc_3_0_, mid2_), 120), mcable_list{{1,0.3,1}, {2,0,0.5}, {3, 0, 0.4}}));

        // Test radius_lt and radius_gt
        EXPECT_TRUE(region_eq(mp, radius_lt(all(), 2), mcable_list{{0,0,0.55}, {1,0,0.325}, {3,0.375,0.75}}));
        EXPECT_TRUE(region_eq(mp, radius_lt(all(), 3), mcable_list{{0,0,1}, {1,0,0.55}, {2,6.0/9.0,1}, {3,0.25,1}}));
        EXPECT_TRUE(region_eq(mp, radius_gt(all(), 2), mcable_list{{0,0.55,1}, {1,0.325,1}, {2,0,1}, {3,0,0.375}, {3,0.75,1}}));
        EXPECT_TRUE(region_eq(mp, radius_gt(all(), 3), mcable_list{{1,0.55,1}, {2,0,6.0/9.0}, {3,0,0.25}}));

        EXPECT_TRUE(region_eq(mp, radius_le(all(), 2), mcable_list{{0,0,0.55}, {1,0,0.325}, {2,1,1}, {3,0.375,0.75}}));
        EXPECT_TRUE(region_eq(mp, radius_le(all(), 3), mcable_list{{0,0,1}, {1,0,0.55}, {2,6.0/9.0,1}, {3,0.25,1}}));
        EXPECT_TRUE(region_eq(mp, radius_ge(all(), 2), mcable_list{{0,0.55,1}, {1,0.325,1}, {2,0,1}, {3,0,0.375}, {3,0.75,1}}));
        EXPECT_TRUE(region_eq(mp, radius_ge(all(), 3), mcable_list{{1,0.55,1}, {2,0,6.0/9.0}, {3,0,0.25}}));

        EXPECT_TRUE(region_eq(mp, radius_lt(reg_a_, 2), mcable_list{{0,0.1,0.4},{3,0.375,0.4}}));
        EXPECT_TRUE(region_eq(mp, radius_gt(reg_a_, 2), mcable_list{{2,0,1},{3,0.1,0.375}}));
        EXPECT_TRUE(region_eq(mp, radius_lt(reg_b_, 2), mcable_list{{0,0.1,0.4}}));
        EXPECT_TRUE(region_eq(mp, radius_gt(reg_c_, 2), mcable_list{{0,0.55,0.7},{2,0,0.5},{3,0.1,0.375},{3,0.9,1}}));
        EXPECT_TRUE(region_eq(mp, radius_gt(reg_d_, 2), mcable_list{{0,0.55,0.7},{2,0,0.5},{3,0.1,0.375},{3,0.75,0.9}}));

        // Test some more interesting intersections and unions.

        //    123456789 123456789
        //   |---------|---------| lhs
        //   |  -----  |   ---   | rhs
        //   |  xxxxx  |   xxx   | rand
        //   |xxxxxxxxx|xxxxxxxxx| ror
        auto lhs  = b13;
        auto rhs  = join(cable(1,.2,.7), cable(3,.3,.6));
        auto rand = cl{{1,.2,.7}, {3,.3,.6}};
        auto ror  = cl{{1,.0,1.}, {3,.0,1.}};
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        //    123456789 123456789
        //   |   ----- | ----    | lhs
        //   |  -----  |   ---   | rhs
        //   |   xxxx  |   xx    | rand
        //   |  xxxxxx | xxxxx   | ror
        lhs  = join(cable(1,.3,.8), cable(3,.1,.5));
        rhs  = join(cable(1,.2,.7), cable(3,.3,.6));
        rand = cl{         {1,.3,.7}, {3,.3,.5}};
        ror  = cl{         {1,.2,.8}, {3,.1,.6}};
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        //    123456789 123456789
        //   | -- -    | --- --- | lhs
        //   |  -----  |   ---   | rhs
        //   |  x x    |   x x   | rand
        //   | xxxxxx  | xxxxxxx | ror
        lhs  = join(cable(1,.1,.3), cable(1,.4,.5), cable(3,.1,.4), cable(3,.5,.9));
        rhs  = join(cable(1,.2,.7), cable(3,.3,.6));
        rand = cl{         {1,.2,.3}, {1,.4,.5}, {3,.3,.4}, {3,.5,.6}};
        ror  = cl{         {1,.1,.7},            {3,.1,.9}};
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        //       b1
        //    123456789
        //   |-----    | lhs
        //   |-----    | rhs
        //   |xxxxx    | rand
        //   |xxxxx    | ror
        lhs  = cable(1,0,.5);
        rhs  = cable(1,0,.5);
        rand = cl{{1,0,.5}};
        ror  = cl{{1,0,.5}};
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        //       b3
        //    123456789
        //   |-----    | lhs
        //   |-----    | rhs
        //   |xxxxx    | rand
        //   |xxxxx    | ror
        lhs  = cable(3,0,.5);
        rhs  = cable(3,0,.5);
        rand = cl{{3,0,.5}};
        ror  = cl{{3,0,.5}};
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        //       b0        b1
        //    123456789 123456789
        //   |xxxxx    |         | lhs
        //   |         |xxxxx    | rhs
        //   x         |         | rand
        //   |xxxxx    |xxxxx    | ror
        lhs  = cable(0,0,.5);
        rhs  = cable(1,0,.5);
        rand = cl{{0,0,0}};
        ror  = cl{{0,0,.5},{1,0,.5}};
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        //       b2        b3
        //    123456789 123456789
        //   |xxxxx    |         | lhs
        //   |         |xxxxx    | rhs
        //   x         |         | rand
        //   |xxxxx    |xxxxx    | ror
        lhs  = cable(2,0,.5);
        rhs  = cable(3,0,.5);
        rand = cl{{1,1,1}};
        ror  = cl{{2,0,.5},{3,0,.5}};
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        //       b0        b3
        //    123456789 123456789
        //   |xxxxx    |xxxxx    | lhs
        //   |xxxxxxx  |xxx      | rhs
        //   |xxxxx    |xxx      | rand
        //   |xxxxxxx  |xxxxx    | ror
        lhs  = join(cable(0,0,.5), cable(3,0,.5));
        rhs  = join(cable(0,0,.7), cable(3,0,.3));
        rand = cl{{0,0,.5},{3,0,.3}};
        ror  = cl{{0,0,.7},{3,0,.5}};
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

        // Assert communtativity
        std::swap(lhs, rhs);
        EXPECT_TRUE(region_eq(mp, intersect(lhs, rhs), rand));
        EXPECT_TRUE(region_eq(mp, join(lhs, rhs), ror));

    }
}
TEST(region, thingify_complex_morphologies) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    {
        pvec parents = {mnpos, 0, 1, 0, 3, 4, 5, 5, 7, 7, 4, 10};
        svec samples = {
                {{  0,  0,  0,  2}, 3}, //0
                {{ 10,  0,  0,  2}, 3}, //1
                {{100,  0,  0,  2}, 3}, //2
                {{  0, 10,  0,  2}, 3}, //3
                {{  0,100,  0,  2}, 3}, //4
                {{100,100,  0,  2}, 3}, //5
                {{200,100,  0,  2}, 3}, //6
                {{100,200,  0,  2}, 3}, //7
                {{200,200,  0,  2}, 3}, //8
                {{100,300,  0,  2}, 3}, //9
                {{  0,200,  0,  2}, 3}, //10
                {{  0,300,  0,  2}, 3}, //11
        };
        sample_tree sm(samples, parents);
        auto m = morphology(sm, false);
        {
            mprovider mp(m);
            using reg::cable;
            using ls::most_distal;
            using ls::most_proximal;

            auto reg_a_ = join(cable(0,0.1,0.4), cable(0,0,0.9), cable(1,0.1,0.4));
            auto reg_b_ = join(cable(0,0.1,0.4), cable(0,0,0.9), cable(1,0.1,0.4), cable(1,0.2,0.5));
            auto reg_c_ = join(cable(0,0.1,0.4), cable(0,0,0.9), cable(1,0.1,0.4), cable(2,0.2,0.5));
            auto reg_d_ = join(cable(2,0,0.9), cable(3,0.1,0.1), cable(4,0.1,0.6));
            auto reg_e_ = join(cable(2,0,0.9), cable(4,0.1,0.1), cable(5,0.1,0.6));
            auto reg_f_ = join(cable(7,0,1), cable(2,0,0.9), cable(4,0.1,0.1), cable(5,0.1,0.6));

            EXPECT_TRUE(mlocationlist_eq(thingify(most_distal(reg_a_), mp), mlocation_list{{0,0.9},{1,0.4}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_distal(reg_b_), mp), mlocation_list{{0,0.9},{1,0.5}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_distal(reg_c_), mp), mlocation_list{{0,0.9},{2,0.5}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_distal(reg_d_), mp), mlocation_list{{3,0.1},{4,0.6}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_distal(reg_e_), mp), mlocation_list{{5,0.6}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_distal(reg_f_), mp), mlocation_list{{5,0.6},{7,1}}));

            EXPECT_TRUE(mlocationlist_eq(thingify(most_proximal(reg_a_), mp), mlocation_list{{0,0}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_proximal(reg_b_), mp), mlocation_list{{0,0}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_proximal(reg_c_), mp), mlocation_list{{0,0}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_proximal(reg_d_), mp), mlocation_list{{1,1}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_proximal(reg_e_), mp), mlocation_list{{1,1}}));
            EXPECT_TRUE(mlocationlist_eq(thingify(most_proximal(reg_f_), mp), mlocation_list{{1,1}}));
        }
    }
    {
        pvec parents = {mnpos, 0, 1, 1, 2, 3, 0, 6, 7, 8, 7};
        svec samples = {
                {{  0, 10, 10,  1}, 1},
                {{  0, 30, 30,  1}, 2},
                {{  0, 60,-20,  1}, 2},
                {{  0, 90, 70,  1}, 2},
                {{  0, 80,-10,  1}, 2},
                {{  0,100,-40,  1}, 2},
                {{  0,-50,-50,  1}, 2},
                {{  0, 20,-30,  2}, 2},
                {{  0, 40,-80,  2}, 2},
                {{  0,-30,-80,  3}, 2},
                {{  0, 90,-70,  5}, 2}
        };
        sample_tree sm(samples, parents);

        // Without spherical root
        mprovider mp(morphology(sm, false));

        using reg::all;
        using reg::nil;
        using reg::z_dist_from_root_lt;
        using reg::z_dist_from_root_le;
        using reg::z_dist_from_root_gt;
        using reg::z_dist_from_root_ge;
        using reg::cable;

        // Test projection
        EXPECT_TRUE(region_eq(mp, z_dist_from_root_lt(0), nil()));
        EXPECT_TRUE(region_eq(mp, z_dist_from_root_gt(0), all()));

        EXPECT_TRUE(region_eq(mp, z_dist_from_root_le(100), all()));
        EXPECT_TRUE(region_eq(mp, z_dist_from_root_gt(100), nil()));

        EXPECT_TRUE(region_eq(mp, z_dist_from_root_le(90), all()));
        EXPECT_TRUE(region_eq(mp, z_dist_from_root_gt(90), nil()));

        EXPECT_TRUE(region_eq(mp, z_dist_from_root_lt(20),
                                  mcable_list{{0,0,1},
                                              {1,0,0.578250901781922829},
                                              {2,0.61499300915417734997,0.8349970039232188642},
                                              {3,0,0.179407353580315756}}));
        EXPECT_TRUE(region_eq(mp, z_dist_from_root_ge(20),
                                  mcable_list{{0,1,1},
                                              {1,0.578250901781922829,1},
                                              {2,0,0.61499300915417734997},
                                              {2,0.8349970039232188642,1},
                                              {3,0.179407353580315756,1},
                                              {4,0,1},
                                              {5,0,1}}));
        EXPECT_TRUE(region_eq(mp, join(z_dist_from_root_lt(20), z_dist_from_root_ge(20)), all()));

        EXPECT_TRUE(region_eq(mp, z_dist_from_root_le(50),
                                  mcable_list{{0,0,1},
                                              {1,0,1},
                                              {2,0,0.2962417607888518767},
                                              {2,0.4499900130773962142,1},
                                              {3,0,0.4485183839507893905},
                                              {3,0.7691110303704736343,1},
                                              {4,0,0.0869615364994152821},
                                              {5,0,0.25}}));
        EXPECT_TRUE(region_eq(mp, z_dist_from_root_gt(50),
                                  mcable_list{{2,0.2962417607888518767,0.4499900130773962142},
                                              {3,0.4485183839507893905,0.7691110303704736343},
                                              {4,0.0869615364994152821,1},
                                              {5,0.25,1}}));

        EXPECT_TRUE(region_eq(mp, join(z_dist_from_root_le(50), z_dist_from_root_gt(50)), all()));
    }
}
