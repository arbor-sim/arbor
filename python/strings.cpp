#include <string>
#include <sstream>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>

#include "event_generator.hpp"
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
      << ", distributed " << (mpi? "MPI": "Local")
      << " ranks " << arb::num_ranks(c)
      << ">";
    return s.str();
}

std::string proc_allocation_string(const arb::proc_allocation& a) {
    std::stringstream s;
    s << "<hardware resource allocation: threads " << a.num_threads << ", gpu ";
    if (a.has_gpu()) {
        s << a.gpu_id;
    }
    else {
        s << "None";
    }
    s << ">";
    return s.str();
}

std::string schedule_explicit_string(const explicit_schedule_shim& e) {
  std::stringstream s;
  s << "<explicit_schedule: times " << e.py_times << " ms>";
  return s.str();
}

std::string schedule_regular_string(const regular_schedule_shim& r) {
  std::stringstream s;
  s << "<regular_schedule: tstart " << r.tstart << " ms"
    << ", dt " << r.dt << " ms"
    << ", tstop " << r.tstop << " ms" << ">";
  return s.str();
}

std::string schedule_poisson_string(const poisson_schedule_shim& p) {
  std::stringstream s;
  s << "<regular_schedule: tstart " << p.tstart << " ms"
    << ", freq " << p.freq << " Hz"
    << ", seed " << p.seed << ">";
  return s.str();
}

} // namespace pyarb
