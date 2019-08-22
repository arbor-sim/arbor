#pragma once

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

namespace pyarb {

struct event_generator_shim {
    arb::cell_member_type target;
    double weight;
    arb::schedule time_sched;

    event_generator_shim(arb::cell_member_type gid, double event_weight, arb::schedule sched):
        target(gid),
        weight(event_weight),
        time_sched(std::move(sched))
    {}
};

} // namespace pyarb
