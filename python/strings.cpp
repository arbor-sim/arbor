#include <sstream>
#include <string>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>

#include "strings.hpp"

namespace pyarb {

std::string cell_member_string(const arb::cell_member_type& m) {
    std::stringstream s;
    s << "<cell_member: gid " << m.gid
      << ", index " << m.index << ">";
    return s.str();
}

std::string context_string(const arb::context& c) {
    std::stringstream s;
    const bool gpu = arb::has_gpu(c);
    const bool mpi = arb::has_mpi(c);
    s << "<context: threads " << arb::num_threads(c)
      << ", gpu " << (gpu? "yes": "None")
      << ", distributed " << (mpi? "MPI": "local")
      << " ranks " << arb::num_ranks(c)
      << ">";
    return s.str();
}

} // namespace pyarb
