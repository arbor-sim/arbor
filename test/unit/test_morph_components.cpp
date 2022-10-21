#include <cmath>
#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/mprovider.hpp>

#include "util/rangeutil.hpp"

#include <gtest/gtest.h>
#include "common_morphologies.hpp"

using namespace arb;

TEST(morph, subbranch_components) {
    using namespace common_morphology;

    const auto& m = m_reg_b1; // simple cable
    mextent ex(mcable_list{mcable{0, 0.125, 0.25}, mcable{0, 0.5, 0.625}, mcable{0, 0.625, 0.75}, mcable{0, 0.875, 1.}});
    auto comps = components(m, ex);

    ASSERT_EQ(3u, comps.size());
    ASSERT_FALSE(util::any_of(comps, [](const auto& x) { return x.empty(); }));

    util::sort_by(comps, [](const mextent& x) { return prox_loc(x.front()); });

    EXPECT_EQ((mcable_list{mcable{0, 0.125, 0.25}}), comps[0].cables());
    EXPECT_EQ((mcable_list{mcable{0, 0.5,   0.75}}), comps[1].cables());
    EXPECT_EQ((mcable_list{mcable{0, 0.875, 1.}}),   comps[2].cables());
}

TEST(morph, subtree_components) {
    using namespace common_morphology;

    const auto& m = m_mlt_b6; // cell with two 'Y's (branches 0, 1, 2 and branches 3, 4, 5) meeting at root.

    // Component semantics has that initial segments of branches from a common fork are _not_ regarded
    // as connected; a final segment of a branch and an initial segment of a child branch _are_ regarded
    // as connected.

    std::pair<mcable_list, std::vector<mcable_list>> test_cases[] =
    {
        // Full cell gives two components (one for each 'Y').
        {
            {mcable{0, 0, 1}, mcable{1, 0, 1}, mcable{2, 0, 1}, mcable{3, 0, 1}, mcable{4, 0, 1}, mcable{5, 0, 1}},
            {
                {mcable{0, 0, 1}, mcable{1, 0, 1}, mcable{2, 0, 1}},
                {mcable{3, 0, 1}, mcable{4, 0, 1}, mcable{5, 0, 1}}
            }
        },

        // Siblings are separated.
        {
            {mcable{1, 0, 1}, mcable{2, 0, 1}, mcable{4, 0, 1}, mcable{5, 0, 1}},
            {
                {mcable{1, 0, 1}},
                {mcable{2, 0, 1}},
                {mcable{4, 0, 1}},
                {mcable{5, 0, 1}}
            }
        },

        // Parent-child branches are connected if they include the fork point.
        {
            {mcable{0, 0.5, 1}, mcable{2, 0, 0.5}, mcable{3, 0.5, 1}, mcable{5, 0.5, 1}},
            {
                {mcable{0, 0.5, 1}, mcable{2, 0, 0.5}},
                {mcable{3, 0.5, 1}},
                {mcable{5, 0.5, 1}}
            }
        },
    };

    for (auto tc: test_cases) {
        auto comps = components(m, mextent(tc.first));
        util::sort_by(comps, [](const mextent& x) { return prox_loc(x.front()); });

        std::vector<mcable_list> result = util::assign_from(util::transform_view(comps, [](const auto& x) { return x.cables(); }));
        EXPECT_EQ(tc.second, result);
    }
}

