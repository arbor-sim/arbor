#include <string>
#include <vector>

#include "blocks.hpp"
#include "io/ostream_wrappers.hpp"

#include "io/pprintf.hpp"

// Pretty-printers for block info.
using namespace io;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    separator s("[", " ");
    for (const auto& x: v) os << s << x;
    os << ']';
    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, Id const& V) {
    os << '(' << V.token << ',' << V.value << ',';
    if(V.units.size()) os << V.units << ',';
    os << ')';
    return os;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, UnitsBlock::units_pair const& p) {
    return os << '(' << p.first << ", " << p.second << ')';
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, IonDep const& I) {
    return os << '('  << I.name << " read=" << I.read << "write=" << I.write << ')';
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, moduleKind const& k) {
    return os << to_string(k);
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, NeuronBlock const& N) {
    return os << fmt::format(FMT_COMPILE("{}\n"
                                         "  kind       : {}\n"
                                         "  name       : {}\n"
                                         "  threadsafe : {}\n"),
                             blue("NeuronBlock"),
                             to_string(N.kind),
                             N.name,
                             N.threadsafe)
        << "  ranges     : " << N.ranges << '\n'
        << "  globals    : " << N.globals << '\n'
        << "  ions       : " << N.ions << '\n';
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, StateBlock const& B) {
    return os << blue("StateBlock") << std::endl
              << "  variables  : " << B.state_variables << std::endl;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, UnitsBlock const& U) {
    return os << blue("UnitsBlock") << std::endl
              << "  aliases    : " << U.unit_aliases << std::endl;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, ParameterBlock const& P) {
    return os << blue("ParameterBlock") << std::endl
              << "  parameters : " << P.parameters << std::endl;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, AssignedBlock const& A) {
    return os << blue("AssignedBlock") << std::endl
              << "  parameters : " << A.parameters << std::endl;
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, WhiteNoiseBlock const& W) {
    return os << blue("WhiteNoiseBlock") << std::endl
              << "  parameters : " << W.parameters << std::endl;
}
