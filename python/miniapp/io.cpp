#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include <tclap/CmdLine.h>
#include <json/json.hpp>

#include <util/meta.hpp>
#include <util/optional.hpp>
#include <util/strprintf.hpp>

#include "io.hpp"

namespace arb {
namespace io {
/// Parse spike times from a stream
/// A single spike per line, trailing whitespace is ignore
/// Throws a usage error when parsing fails
///
/// Returns a vector of time_type

std::vector<time_type> parse_spike_times_from_stream(std::ifstream & fid) {
    std::vector<time_type> times;
    std::string line;
    while (std::getline(fid, line)) {
        std::stringstream s(line);

        time_type t;
        s >> t >> std::ws;

        if (!s || s.peek() != EOF) {
            throw std::runtime_error( util::strprintf(
                    "Unable to parse spike file on line %d: \"%s\"\n",
                    times.size(), line));
        }

        times.push_back(t);
    }

    return times;
}

/// Parse spike times from a file supplied in path
/// A single spike per line, trailing white space is ignored
/// Throws a usage error when opening file or parsing fails
///
/// Returns a vector of time_type

std::vector<time_type> get_parsed_spike_times_from_path(arb::util::path path) {
    std::ifstream fid(path);
    if (!fid) {
        throw std::runtime_error(util::strprintf(
            "Unable to parse spike file: \"%s\"\n", path.c_str()));
    }

    return parse_spike_times_from_stream(fid);
}

std::ostream& operator<<(std::ostream& o, const io::options& options) {
    o << "simulation options:\n";
    o << "  cells                : " << options.cells << "\n";
    o << "  compartments/segment : " << options.compartments_per_segment << "\n";
    o << "  synapses/cell        : " << options.synapses_per_cell << "\n";
    o << "  simulation time      : " << options.tfinal << "\n";
    o << "  dt                   : " << options.dt << "\n";
    o << "  binning dt           : " << options.bin_dt << "\n";
    o << "  binning policy       : " <<
        (options.bin_dt==0? "none": options.bin_regular? "regular": "following") << "\n";
    o << "  all to all network   : " << (options.all_to_all ? "yes" : "no") << "\n";
    o << "  ring network         : " << (options.ring ? "yes" : "no") << "\n";
    o << "  sample dt            : " << options.sample_dt << "\n";
    o << "  probe ratio          : " << options.probe_ratio << "\n";
    o << "  probe soma only      : " << (options.probe_soma_only ? "yes" : "no") << "\n";
    o << "  trace prefix         : " << options.trace_prefix << "\n";
    o << "  trace max gid        : ";
    if (options.trace_max_gid) {
       o << *options.trace_max_gid;
    }
    o << "\n";
    o << "  trace format         : " << options.trace_format << "\n";
    o << "  morphologies         : ";
    if (options.morphologies) {
       o << *options.morphologies;
    }
    o << "\n";
    o << "  morphology r-r       : " << (options.morph_rr ? "yes" : "no") << "\n";
    o << "  report compartments  : " << (options.report_compartments ? "yes" : "no") << "\n";

    return o;
}

} // namespace io
} // namespace arb
