#pragma once

#include <arbor/common_types.hpp>

#include "util/range.hpp"
#include "backends/threshold_crossing.hpp"
#include "execution_context.hpp"

namespace arb {

struct fvm_integration_result {
    util::range<const threshold_crossing*> crossings;
    util::range<const arb_value_type*> sample_time;
    util::range<const arb_value_type*> sample_value;
};

struct fvm_detector_info {
    arb_size_type count = 0;
    std::vector<arb_index_type> cv;
    std::vector<arb_value_type> threshold;
    execution_context ctx;
};

}
