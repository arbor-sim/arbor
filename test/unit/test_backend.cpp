#include <arbor/common_types.hpp>
#include <arbor/version.hpp>

#include "fvm_lowered_cell.hpp"

#include "../gtest.h"

using namespace arb;

TEST(backends, gpu_test) {
#ifdef ARB_GPU_ENABLED
    EXPECT_NO_THROW(make_fvm_lowered_cell(backend_kind::gpu));
#else
    EXPECT_ANY_THROW(make_fvm_lowered_cell(backend_kind::gpu));
#endif
}
