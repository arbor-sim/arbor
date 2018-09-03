#include <string>
#include <sstream>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>

#include "strings.hpp"

namespace arb {
namespace py {

std::string cell_member_string(const arb::cell_member_type& m) {
    std::stringstream s;
    s << "<cell_member: gid " << m.gid << ", index " << m.index << ">";
    return s.str();
}

std::string local_resources_string(const arb::local_resources& r) {
    std::stringstream s;
    s << "<local_resources: threads " << r.num_threads << ", gpus " << r.num_gpus << ">";
    return s.str();
}

std::string proc_allocation_string(const arb::proc_allocation& a) {
    std::stringstream s;
    s << "<local_resources: threads " << a.num_threads
      << ", gpu " << (a.has_gpu()? "yes": "None");
    if (a.has_gpu()) {
        s << " (id " << a.gpu_id << ")";
    }
    s << ">";
    return s.str();
}

std::string context_string(const arb::context& c) {
    std::stringstream s;
    const bool gpu = arb::has_gpu(c);
    const bool mpi = arb::has_mpi(c);
    s << "<context: threads " << arb::num_threads(c)
      << ", gpu " << (gpu? "yes": "None")
      << ", distributed " << (mpi? "MPI": "Local")
      << " ranks " << arb::num_ranks(c)
      << ">";
    return s.str();
}

} // namespace py
} // namespace arb
