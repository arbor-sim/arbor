#ifndef ARBOR_MINIAPP_BASE
#define ARBOR_MINIAPP_BASE

#include <vector>
#include <string>

#include "util/optional.hpp"

namespace arb {
namespace io {

struct Args {
    // think...
    std::vector<std::string> args;
    int argc;
    char** argv;
    std::vector<char*> storage;

    Args() {
        set(std::vector<std::string>());
    }

    void set(const std::vector<std::string>& args_)
    {
        args = args_;
        storage.clear();
        
        for (auto&& arg: args) {
            storage.push_back((char*)arg.c_str());
        }
        storage.push_back(nullptr);
        argc = storage.size()-1;
        argv = (char**) storage.data();
    }
};

struct Options {
    // command line args
    Args args;
    
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

std::ostream& operator<<(std::ostream& o, const Options& opt);


// calls from python
class OptionsInterface: private Options {
public:
    Options& get_options() {
        return *this;
    }
    
    void set_args(std::vector<std::string> args_) {
        args.set(args_);
    }
    void set_cells(uint32_t _cells) {
        cells = _cells;
    }
    void set_synapses_per_cell(uint32_t _synapses_per_cell) {
        synapses_per_cell = _synapses_per_cell;
    }
    void set_syn_type(std::string _syn_type) {
        syn_type = _syn_type;
    }
    void set_compartments_per_segment(uint32_t _compartments_per_segment) {
        compartments_per_segment = _compartments_per_segment;
    }
    void set_morphologies(std::string _morphologies) {
        morphologies = _morphologies;
    }
    void set_morph_rr(bool _morph_rr) {
        morph_rr = _morph_rr;
    }
    void set_all_to_all(bool _all_to_all) {
        all_to_all = _all_to_all;
    }
    void set_ring(bool _ring) {
        ring = _ring;
    }
    void set_tfinal(double _tfinal) {
        tfinal = _tfinal;
    }
    void set_dt(double _dt) {
        dt = _dt;
    }
    void set_bin_regular(bool _bin_regular) {
        bin_regular = _bin_regular;
    }
    void set_bin_dt(double _bin_dt) {
        bin_dt = _bin_dt;
    }
    void set_sample_dt(double _sample_dt) {
        sample_dt = _sample_dt;
    }
    void set_probe_soma_only(bool _probe_soma_only) {
        probe_soma_only = _probe_soma_only;
    }
    void set_probe_ratio(double _probe_ratio) {
        probe_ratio = _probe_ratio;
    }
    void set_trace_prefix(std::string _trace_prefix) {
        trace_prefix = _trace_prefix;
    }
    void set_trace_max_gid(unsigned _trace_max_gid) {
        trace_max_gid = _trace_max_gid;
    }
    void set_trace_format(std::string _trace_format) {
        trace_format = _trace_format;
    }
    void set_spike_file_output(bool _spike_file_output) {
        spike_file_output = _spike_file_output;
    }
    void set_single_file_per_rank(bool _single_file_per_rank) {
        single_file_per_rank = _single_file_per_rank;
    }
    void set_over_write(bool _over_write) {
        over_write = _over_write;
    }
    void set_output_path(std::string _output_path) {
        output_path = _output_path;
    }
    void set_file_name(std::string _file_name) {
        file_name = _file_name;
    }
    void set_file_extension(std::string _file_extension) {
        file_extension = _file_extension;
    }
    void set_spike_file_input(bool _spike_file_input) {
        spike_file_input = _spike_file_input;
    }
    void set_input_spike_path(std::string _input_spike_path) {
        input_spike_path = _input_spike_path;
    }
    void set_dry_run_ranks(int _dry_run_ranks) {
        dry_run_ranks = _dry_run_ranks;
    }
    void set_profile_only_zero(bool _profile_only_zero) {
        profile_only_zero = _profile_only_zero;
    }
    void set_report_compartments(bool _report_compartments) {
        report_compartments = _report_compartments;
    }
    void set_verbose(bool _verbose) {
        verbose = _verbose;
    }
};

int miniapp(const Options&);

#endif //ARBOR_MINIAPP_BASE
