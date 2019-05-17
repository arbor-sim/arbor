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

struct sample_record {
    time_type time;
    util::any_ptr data;
};

using sampler_function = std::function<void (cell_member_type, probe_tag, std::size_t, const sample_record*)>;

using sampler_association_handle = std::size_t;

enum class sampling_policy {
    lax,
    // interpolated, // placeholder: unsupported
    // exact         // placeholder: unsupported
};

} // namespace arb
