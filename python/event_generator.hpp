#pragma once

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

namespace pyarb {

struct event_generator_shim {
    arb::cell_local_label_type target;
    double weight;
    arb::schedule time_sched;

    event_generator_shim(arb::cell_local_label_type event_target, double event_weight, arb::schedule sched):
        target(std::move(event_target)),
        weight(event_weight),
        time_sched(std::move(sched))
    {}
};

} // namespace pyarb
