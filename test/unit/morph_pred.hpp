#pragma once

// Predicates for morphology testing.

#include "../gtest.h"

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>

#include "util/span.hpp"

namespace testing {

inline ::testing::AssertionResult mlocation_eq(arb::mlocation a, arb::mlocation b) {
    if (a.branch!=b.branch) {
        return ::testing::AssertionFailure()
                << "cables " << a << " and " << b << " differ";
    }

    using FP = testing::internal::FloatingPoint<double>;
    if (FP(a.pos).AlmostEquals(FP(b.pos))) {
        return ::testing::AssertionSuccess();
    }
    else {
        return ::testing::AssertionFailure()
                << "mlocations " << a << " and " << b << " differ";
    }
}

inline ::testing::AssertionResult cable_eq(arb::mcable a, arb::mcable b) {
    if (a.branch!=b.branch) {
        return ::testing::AssertionFailure()
            << "cables " << a << " and " << b << " differ";
    }

    using FP = testing::internal::FloatingPoint<double>;
    if (FP(a.prox_pos).AlmostEquals(FP(b.prox_pos)) && FP(a.dist_pos).AlmostEquals(FP(b.dist_pos))) {
        return ::testing::AssertionSuccess();
    }
    else {
        return ::testing::AssertionFailure()
            << "cables " << a << " and " << b << " differ";
    }
}

inline ::testing::AssertionResult cablelist_eq(const arb::mcable_list& as, const arb::mcable_list& bs) {
    if (as.size()!=bs.size()) {
        return ::testing::AssertionFailure()
            << "cablelists " << as << " and " << bs << " differ";
    }

    for (auto i: arb::util::count_along(as)) {
        auto result = cable_eq(as[i], bs[i]);
        if (!result) return ::testing::AssertionFailure()
            << "cablelists " << as << " and " << bs << " differ";
    }
    return ::testing::AssertionSuccess();
}

inline ::testing::AssertionResult extent_eq(const arb::mextent& xa, const arb::mextent& xb) {
    return cablelist_eq(xa.cables(), xb.cables());
}

inline ::testing::AssertionResult region_eq(const arb::mprovider& p, arb::region a, arb::region b) {
    return extent_eq(thingify(a, p), thingify(b, p));
}

inline ::testing::AssertionResult mlocationlist_eq(const arb::mlocation_list& as, const arb::mlocation_list& bs) {
    if (as.size()!=bs.size()) {
        return ::testing::AssertionFailure()
                << "cablelists " << as << " and " << bs << " differ";
    }

    for (auto i: arb::util::count_along(as)) {
        auto result = mlocation_eq(as[i], bs[i]);
        if (!result) return ::testing::AssertionFailure()
                    << "mlocation lists " << as << " and " << bs << " differ";
    }
    return ::testing::AssertionSuccess();
}

} // namespace testing

