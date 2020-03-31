#pragma once

#include <memory>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/recipe.hpp>

#include "backends/event.hpp"
#include "backends/threshold_crossing.hpp"
#include "execution_context.hpp"
#include "sampler_map.hpp"
#include "util/range.hpp"

namespace arb {

struct fvm_integration_result {
    util::range<const threshold_crossing*> crossings;
    util::range<const fvm_value_type*> sample_time;
    util::range<const fvm_value_type*> sample_value;
};

// A sample for a probe may be derived from multiple 'raw' sampled
// values from the backend.
//
// While supported probes are at this point all simple scalar values,
// fvm_probe_info will be the class that represents the mapping
// between a single sample result and the back-end raw probe handles.

struct fvm_probe_info {
    // nullptr => nothing to probe
    probe_handle raw_handle = nullptr;
};

// Common base class for FVM implementation on host or gpu back-end.

struct fvm_lowered_cell {
    virtual void reset() = 0;

    virtual void initialize(
        const std::vector<cell_gid_type>& gids,
        const recipe& rec,
        std::vector<fvm_index_type>& cell_to_intdom,
        std::vector<target_handle>& target_handles,
        probe_association_map<probe_handle>& probe_map) = 0;

    virtual fvm_integration_result integrate(
        fvm_value_type tfinal,
        fvm_value_type max_dt,
        std::vector<deliverable_event> staged_events,
        std::vector<sample_event> staged_samples) = 0;

    virtual fvm_value_type time() const = 0;

    virtual ~fvm_lowered_cell() {}
};

using fvm_lowered_cell_ptr = std::unique_ptr<fvm_lowered_cell>;

fvm_lowered_cell_ptr make_fvm_lowered_cell(backend_kind p, const execution_context& ctx);

} // namespace arb
