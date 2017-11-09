#include <iostream>

#include <common_types.hpp>

std::ostream& operator<<(std::ostream& O, arb::cell_member_type m) {
    return O << m.gid << ':' << m.index;
}

