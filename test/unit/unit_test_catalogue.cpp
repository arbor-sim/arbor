#include <arbor/mechcat.hpp>
#include <arbor/version.hpp>

#include "testing_catalogue.hpp"

#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#endif
#include "backends/multicore/fvm.hpp"

#include "unit_test_catalogue.hpp"

#include <gtest/gtest.h>

#define ADD_MECH(c, x) do {                                             \
    auto mech = make_testing_##x();                                     \
    c.add(#x, mech.type());                                             \
    c.register_implementation(#x, std::make_unique<arb::mechanism>(mech.type(), *mech.i_cpu())); \
    if (mech.i_gpu()) {                                                 \
        c.register_implementation(#x, std::make_unique<arb::mechanism>(mech.type(), *mech.i_gpu())); \
    }                                                                   \
    } while (false)

arb::mechanism_catalogue make_unit_test_catalogue(const arb::mechanism_catalogue& from) {
    auto result = from;
    result.import(arb::global_testing_catalogue(), "");
    return result;
}

