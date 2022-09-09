#include <iostream>

#include <arbor/common_types.hpp>

namespace arb {

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, lid_selection_policy policy) {
    switch (policy) {
    case lid_selection_policy::round_robin:
        return o << "round_robin";
	case lid_selection_policy::round_robin_halt:
        return o << "round_robin_halt";
    case lid_selection_policy::assert_univalent:
        return o << "univalent";
    }
    return o;
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, arb::cell_member_type m) {
    return o << m.gid << ':' << m.index;
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, arb::cell_kind k) {
    o << "cell_kind::";
    switch (k) {
    case arb::cell_kind::spike_source:
        return o << "spike_source";
    case arb::cell_kind::cable:
        return o << "cable";
    case arb::cell_kind::lif:
        return o << "lif";
    case arb::cell_kind::benchmark:
        return o << "benchmark_cell";
    }
    return o;
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, arb::backend_kind k) {
    o << "backend_kind::";
    switch (k) {
    case arb::backend_kind::multicore:
        return o << "multicore";
    case arb::backend_kind::gpu:
        return o << "gpu";
    }
    return o;
}

}
