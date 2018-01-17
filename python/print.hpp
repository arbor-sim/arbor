#pragma once

#include <cxxabi.h>

/*
 * Utility code for generating string representations of types for use in Python
 * wrappers.
 */

#include <string>
#include <sstream>

#include <domain_decomposition.hpp>
#include <hardware/node_info.hpp>
#include <profiling/meter_manager.hpp>
#include <recipe.hpp>
#include <rss_cell.hpp>

inline
std::string demangle(const char* mangled) {
    int status;
    char *demangled = abi::__cxa_demangle(mangled, 0, 0, &status);
    std::string name;
    if (demangled!=nullptr) {
        name = demangled;
        std::free(demangled);
    }
    else {
        name = "<unknown type>";
    }
    return name;
}

inline
std::string cell_member_string(arb::cell_member_type m) {
    std::stringstream s;
    s << "<cell_member: gid " << m.gid << ", index " << m.index << ">";
    return s.str();
}

inline
std::string spike_string(arb::spike sp) {
    std::stringstream s;
    s << "<spike: " << sp.source.gid << ":" << sp.source.index
      << " @ " << sp.time << " ms>";
    return s.str();
}

inline
std::string any_string(const arb::util::any& a) {
    if (a.has_value()) {
        return std::string("<any: ") + demangle(a.type().name()) + ">";
    }
    return "<any: empty>";
}

inline
std::string meter_report_string(const arb::util::meter_report& r) {
    std::stringstream s;
    s << "meter report:\n";
    s << r;
    return s.str();
}

inline
std::string rss_cell_string(const arb::rss_cell& c) {
    return "<regular spiking cell: "
         "from " + std::to_string(c.start_time)
        +" to "  + std::to_string(c.stop_time)
        +" by "  + std::to_string(c.period) +">";
}

inline
const char* cell_kind_string(arb::cell_kind k) {
    switch (k) {
    case arb::cell_kind::regular_spike_source:
        return "regular spike source";
    case arb::cell_kind::cable1d_neuron:
        return "cable 1d";
    case arb::cell_kind::data_spike_source:
        return "data spike source";
    }
    return "";
}

inline
const char* backend_kind_string(arb::backend_kind k) {
    switch (k) {
    case arb::backend_kind::gpu:
        return "gpu";
    case arb::backend_kind::multicore:
        return "multicore";
    }
    return "";
}

inline
std::string group_description_string(const arb::group_description& g) {
    std::stringstream s;
    const auto ncells = g.gids.size();
    s << "<group_description: "
      << ncells << " " << cell_kind_string(g.kind)
      << " cells on " << backend_kind_string(g.backend)
      << " gids {";
    if (ncells<5) {
        for (auto i: g.gids) {
            s << i << " ";
        }
    }
    else {
        s << g.gids[0] << " " << g.gids[1] << " " << g.gids[2] << " ... " << g.gids.back();
    }
    s << "}";

    return s.str();
}

inline
std::string domain_decomposition_string(const arb::domain_decomposition& d) {
    std::stringstream s;
    s << "<domain_description:"
      << " domain " << d.domain_id << " ∈ [0," << d.num_domains << "); "
      << d.num_local_cells << " local cells in "
      << d.groups.size() << " cell groups; "
      << d.num_global_cells << " global cells>";

    return s.str();
}

inline
std::string node_info_string(const arb::hw::node_info& nd) {
    std::stringstream s;
    s << "<node_info: " << nd.num_cpu_cores << " cpus; " << nd.num_gpus << " gpus>";

    return s.str();
}
