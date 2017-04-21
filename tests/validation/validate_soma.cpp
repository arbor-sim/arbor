#include "validate_soma.hpp"

#include "../gtest.h"


const auto backend = nest::mc::backend_policy::use_multicore;

TEST(soma, numeric_ref) {
    validate_soma(backend);
}
