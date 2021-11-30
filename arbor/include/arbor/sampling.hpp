#pragma once

#include <cstddef>
#include <functional>

#include <arbor/common_types.hpp>
#include <arbor/util/any_ptr.hpp>

namespace arb {

using cell_member_predicate = std::function<bool (cell_member_type)>;

static cell_member_predicate all_probes = [](cell_member_type pid) { return true; };

inline cell_member_predicate one_probe(cell_member_type pid) {
    return [pid](cell_member_type x) { return pid==x; };
}

// Probe-specific metadata is provided by cell group implementations.
//
// User code is responsible for correctly determining the metadata type,
// but the value of that metadata must be sufficient to determine the
// correct interpretation of sample data provided to sampler callbacks.

struct probe_metadata {
    cell_member_type id; // probe id
    probe_tag tag;       // probe tag associated with id
    unsigned index;      // index of probe source within those supplied by probe id
    util::any_ptr meta;  // probe-specific metadata
};

struct sample_record {
    time_type time;
    util::any_ptr data;
};

using sampler_function = std::function<
    void (probe_metadata,
          std::size_t,          // number of sample records
          const sample_record*  // pointer to first sample record
         )>;

using sampler_association_handle = std::size_t;

enum class sampling_policy {
    lax,
    exact
};

} // namespace arb
