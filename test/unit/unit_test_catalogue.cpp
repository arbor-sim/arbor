#include <arbor/mechcat.hpp>
#include <arbor/version.hpp>

#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#endif
#include "backends/multicore/fvm.hpp"

#include "unit_test_catalogue.hpp"
#include "mechanisms/celsius_test.hpp"
#include "mechanisms/test_cl_valence.hpp"
#include "mechanisms/test_ca_read_valence.hpp"

#include "../gtest.h"

#ifndef ARB_GPU_ENABLED
#define ADD_MECH(c, x)\
c.add(#x, testing::mechanism_##x##_info());\
c.register_implementation(#x, testing::make_mechanism_##x<multicore::backend>());
#else
#define ADD_MECH(c, x)\
c.add(#x, testing::mechanism_##x##_info());\
c.register_implementation(#x, testing::make_mechanism_##x<multicore::backend>());\
c.register_implementation(#x, testing::make_mechanism_##x<gpu::backend>());
#endif

using namespace arb;

mechanism_catalogue make_unit_test_catalogue() {
    mechanism_catalogue cat;

    ADD_MECH(cat, celsius_test)
    ADD_MECH(cat, test_cl_valence)
    ADD_MECH(cat, test_ca_read_valence)

    return cat;
}

