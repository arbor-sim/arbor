#pragma once

// Helper classes for managing sampler/schedule associations in
// cell group classes (see sampling_api doc).

#include <unordered_map>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/schedule.hpp>

namespace arb {

// An association between a samplers, schedule, and set of probe ids, as provided
// to e.g. `model::add_sampler()`.

struct sampler_association {
    schedule sched;
    sampler_function sampler;
    std::vector<cell_member_type> probeset_ids;
    sampling_policy policy;
};

using sampler_association_map = std::unordered_map<sampler_association_handle, sampler_association>;

} // namespace arb
