#include <arbor/mechcat.hpp>
#include <arbor/version.hpp>

#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#endif
#include "backends/multicore/fvm.hpp"

#include "unit_test_catalogue.hpp"
#include "mechanisms/celsius_test.hpp"

#include "../gtest.h"

using namespace arb;

mechanism_catalogue make_unit_test_catalogue() {
    mechanism_catalogue cat;

    cat.add("celsius_test", mechanism_celsius_test_info());

    cat.register_implementation("celsius_test", make_mechanism_celsius_test<multicore::backend>());
#ifdef ARB_GPU_ENABLED
    cat.register_implementation("celsius_test", make_mechanism_celsius_test<gpu::backend>());
#endif

    return cat;
}

