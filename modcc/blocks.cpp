#include <string>
#include <vector>

#include "blocks.hpp"
#include "io/pprintf.hpp"
#include "io/ostream_wrappers.hpp"

// Pretty-printers for block info.

using namespace io;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    separator s("[", " ");
    for (auto& x: v) os << s << x;
    os << ']';

    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, Id const& V) {
    if(V.units.size())
        os << "(" << V.token << "," << V.value << "," << V.units << ")";
    else
        os << "(" << V.token << "," << V.value << ",)";

    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, UnitsBlock::units_pair const& p) {
    return os << "(" << p.first << ", " << p.second << ")";
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, IonDep const& I) {
    return os << "(" << I.name << ": read " << I.read << " write " << I.write << ")";
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, moduleKind const& k) {
    switch (k) {
        case moduleKind::density:  os << "density";  break;
        case moduleKind::voltage:  os << "voltage";  break;
        case moduleKind::point:    os << "point";    break;
        case moduleKind::revpot:   os << "revpot";   break;
        case moduleKind::junction: os << "junction"; break;
        default:                   os << "???";      break;
    }
    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, NeuronBlock const& N) {
    os << blue("NeuronBlock")     << std::endl;
    os << "  kind       : " << N.kind  << std::endl;
    os << "  name       : " << N.name  << std::endl;
    os << "  threadsafe : " << (N.threadsafe ? "yes" : "no") << std::endl;
    os << "  ranges     : " << N.ranges  << std::endl;
    os << "  globals    : " << N.globals << std::endl;
    os << "  ions       : " << N.ions    << std::endl;

    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, StateBlock const& B) {
    os << blue("StateBlock")      << std::endl;
    return os << "  variables  : " << B.state_variables << std::endl;

}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, UnitsBlock const& U) {
    os << blue("UnitsBlock")      << std::endl;
    os << "  aliases    : "  << U.unit_aliases << std::endl;

    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, ParameterBlock const& P) {
    os << blue("ParameterBlock")   << std::endl;
    os << "  parameters : "  << P.parameters << std::endl;

    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, AssignedBlock const& A) {
    os << blue("AssignedBlock")   << std::endl;
    os << "  parameters : "  << A.parameters << std::endl;

    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, WhiteNoiseBlock const& W) {
    os << blue("WhiteNoiseBlock")   << std::endl;
    os << "  parameters : "  << W.parameters << std::endl;

    return os;
}
