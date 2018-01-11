#pragma once

#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <common_types.hpp>
#include <util/optional.hpp>
#include <util/path.hpp>

namespace arb {
namespace io {

// Holds the options for a simulation run.
// Default constructor gives default options.

struct cl_options {
    // Cell parameters:
    uint32_t cells = 1000;
    uint32_t synapses_per_cell = 500;
    std::string syn_type = "expsyn";
    uint32_t compartments_per_segment = 100;
    util::optional<std::string> morphologies;
    bool morph_rr = false; // False => pick morphologies randomly, true => pick morphologies round-robin.

    // Network type (default is rgraph):
    bool all_to_all = false;
    bool ring = false;

    // Simulation running parameters:
    double tfinal = 100.;
    double dt = 0.025;
    bool bin_regular = false; // False => use 'following' instead of 'regular'.
    double bin_dt = 0.0025;   // 0 => no binning.

    // Probe/sampling specification.
    double sample_dt = 0.1;
    bool probe_soma_only = false;
    double probe_ratio = 0;  // Proportion of cells to probe.
    std::string trace_prefix = "trace_";
    util::optional<unsigned> trace_max_gid; // Only make traces up to this gid.
    std::string trace_format = "json"; // Support only 'json' and 'csv'.

    // Parameters for spike output.
    bool spike_file_output = false;
    bool single_file_per_rank = false;
    bool over_write = true;
    std::string output_path = "./";
    std::string file_name = "spikes";
    std::string file_extension = "gdf";

    // Parameters for spike input.
    bool spike_file_input = false;
    std::string input_spike_path;  // Path to file with spikes

    // Dry run parameters (pertinent only when built with 'dryrun' distrib model).
    int dry_run_ranks = 1;

    // Turn on/off profiling output for all ranks.
    bool profile_only_zero = false;

    // Report (inefficiently) on number of cell compartments in sim.
    bool report_compartments = false;

    // Be more verbose with informational messages.
    bool verbose = false;
};

class usage_error: public std::runtime_error {
public:
    template <typename S>
    usage_error(S&& whatmsg): std::runtime_error(std::forward<S>(whatmsg)) {}
};

class model_description_error: public std::runtime_error {
public:
    template <typename S>
    model_description_error(S&& whatmsg): std::runtime_error(std::forward<S>(whatmsg)) {}
};

std::ostream& operator<<(std::ostream& o, const cl_options& opt);

cl_options read_options(int argc, char** argv, bool allow_write = true);

/// Helper function for loading a vector of spike times from file
/// Spike times are expected to be in milli seconds floating points
/// On spike-time per line

std::vector<time_type>  get_parsed_spike_times_from_path(arb::util::path path);

} // namespace io
} // namespace arb
