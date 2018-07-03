#include <iostream>

#include <arbor/common_types.hpp>

std::ostream& operator<<(std::ostream& O, arb::cell_member_type m) {
    return O << m.gid << ':' << m.index;
}

std::ostream& operator<<(std::ostream& o, arb::cell_kind k) {
    o << "cell_kind::";
    switch (k) {
    case arb::cell_kind::spike_source:
        return o << "spike_source";
    case arb::cell_kind::cable1d_neuron:
        return o << "cable1d_neuron";
    case arb::cell_kind::lif_neuron:
        return o << "lif_neuron";
    case arb::cell_kind::benchmark:
        return o << "benchmark_cell";
    }
    return o;
}

std::ostream& operator<<(std::ostream& o, arb::backend_kind k) {
    o << "backend_kind::";
    switch (k) {
    case arb::backend_kind::multicore:
        return o << "multicore";
    case arb::backend_kind::gpu:
        return o << "gpu";
    }
    return o;
}
