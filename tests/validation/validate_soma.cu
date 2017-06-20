#include "validate_soma.hpp"

#include "../gtest.h"

const auto backend = nest::mc::backend_policy::prefer_gpu;

TEST(soma, numeric_ref) {
    validate_soma(backend);
}
