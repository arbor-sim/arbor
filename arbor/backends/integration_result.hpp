#pragma once

#include <arbor/common_types.hpp>

#include "util/range.hpp"
#include "threshold_crossing.hpp"

namespace arb {

struct fvm_integration_result {
    util::range<const threshold_crossing*> crossings;
    util::range<const arb_value_type*> sample_time;
    util::range<const arb_value_type*> sample_value;
};

}
