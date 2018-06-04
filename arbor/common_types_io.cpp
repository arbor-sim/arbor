#include <iostream>

#include <common_types.hpp>

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
    }
    return o;
}

