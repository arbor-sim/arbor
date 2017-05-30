#pragma once

#include <cell.hpp>
#include <morphology.hpp>
#include <segment.hpp>

namespace nest {
namespace mc {

enum class probeKind {
    membrane_voltage,
    membrane_current
};

struct probe_record {
    cell_member_type id;
    segment_location location;
    probeKind kind;
};

} // namespace mc
} // namespace nest
