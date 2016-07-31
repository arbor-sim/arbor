#include <iostream>

#include <catypes.hpp>

std::ostream &operator<<(std::ostream& O, nest::mc::cell_member_type m) {
    return O << m.gid << ':' << m.index;
}

