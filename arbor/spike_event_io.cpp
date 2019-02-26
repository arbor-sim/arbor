#include <iostream>

#include <arbor/spike_event.hpp>

namespace arb {

std::ostream& operator<<(std::ostream& o, const spike_event& ev) {
     return o << "E[tgt " << ev.target << ", t " << ev.time << ", w " << ev.weight << "]";
}

} // namespace arb
