#include <string>
#include <sstream>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/recipe.hpp>

#include "strings.hpp"

namespace pyarb {

std::string cell_member_string(const arb::cell_member_type& m) {
    std::stringstream s;
    s << "<cell_member: gid " << m.gid << ", index " << m.index << ">";
    return s.str();
}

std::string connection_string(const arb::cell_connection& c) {
    std::stringstream s;
    s << "<connection: (" << c.source.gid << "," << c.source.index << ")"
      << " -> (" << c.dest.gid << "," << c.dest.index << ")"
      << " del " << c.delay << ", wgt " << c.weight << ">";
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

std::string segment_location_string(const arb::segment_location& loc) {
    std::stringstream s;
    s << "<location: seg " << loc.segment << " pos " << loc.position << ">";
    return s.str();
}

std::string group_description_string(const arb::group_description& g) {
    std::stringstream s;
    const auto ncells = g.gids.size();
    s << "<cell group: "
      << ncells << " " << g.kind
      << " on " << g.backend;
    if (ncells==1) {
        s << " gid " << g.gids[0];
    }

    else if (ncells<5) {
        s << ", gids {";
        for (auto i: g.gids) {
            s << i << " ";
        }
        s << "}";
    }
    else {
        s << ", gids {";
        s << g.gids[0] << " " << g.gids[1] << " " << g.gids[2] << " ... " << g.gids.back();
        s << "}";
    }

    s << ">";

    return s.str();
}

std::string cell_string(const arb::mc_cell& c) {
    std::stringstream s;
    auto pick = [](std::stringstream& s, unsigned i, const char* a, const char* b) {
        s << i << " " << (i==1? a: b);
    };

    s << "<cell: ";
    pick(s, c.num_segments(), "sections, ", "sections, ");
    pick(s, c.synapses().size(), "synapse, ", "synapses, ");
    pick(s, c.stimuli().size(), "stimulus, ", "stimuli, ");
    pick(s, c.detectors().size(), "detector", "detectors");
    s << ">";

    return s.str();
}

std::string spike_string(const arb::spike& sp) {
    std::stringstream s;


    return s.str();
}

} // namespace pyarb
