#include <type_traits>

#include <backends.hpp>
#include <fvm_lowered_cell.hpp>
#include <util/config.hpp>

#include "../gtest.h"

using namespace arb;

TEST(backends, gpu_test) {
    if (!arb::config::has_cuda) {
        EXPECT_ANY_THROW(make_fvm_lowered_cell(backend_kind::gpu));
    }
    else {
        EXPECT_NO_THROW(make_fvm_lowered_cell(backend_kind::gpu));
    }
}
