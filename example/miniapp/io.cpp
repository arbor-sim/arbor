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

// Let TCLAP understand value arguments that are of an optional type.

namespace TCLAP {
    template <typename V>
    struct ArgTraits<arb::util::optional<V>> {
        using ValueCategory = ValueLike;
    };
} // namespace TCLAP

namespace arb {

namespace util {
    // Using static here because we do not want external linkage for this operator.
    template <typename V>
    static std::istream& operator>>(std::istream& I, optional<V>& v) {
        V u;
        if (I >> u) {
            v = u;
        }
        return I;
    }
}

namespace io {

// Override annoying parameters listed back-to-front behaviour.
//
// TCLAP argument creation _prepends_ its arguments to the internal
// list (_argList), where standard options --help etc. are already
// pre-inserted.
//
// reorder_arguments() reverses the arguments to restore ordering,
// and moves the standard options to the end.
class CustomCmdLine: public TCLAP::CmdLine {
public:
    CustomCmdLine(const std::string &message, const std::string &version = "none"):
        TCLAP::CmdLine(message, ' ', version, true)
    {}

    void reorder_arguments() {
        _argList.reverse();
        for (auto opt: {"help", "version", "ignore_rest"}) {
            auto i = std::find_if(
                _argList.begin(), _argList.end(),
                [&opt](TCLAP::Arg* a) { return a->getName()==opt; });

            if (i!=_argList.end()) {
                auto a = *i;
                _argList.erase(i);
                _argList.push_back(a);
            }
        }
    }
};

// Update an option value from command line argument if set.
template <
    typename T,
    typename Arg,
    typename = util::enable_if_t<std::is_base_of<TCLAP::Arg, Arg>::value>
>
static void update_option(T& opt, Arg& arg) {
    if (arg.isSet()) {
        opt = arg.getValue();
    }
}

// Update an option value from json object if key present.
template <typename T>
static void update_option(T& opt, const nlohmann::json& j, const std::string& key) {
    if (j.count(key)) {
        opt = j[key];
    }
}

// --- special case for string due to ambiguous overloading in json library.
static void update_option(std::string& opt, const nlohmann::json& j, const std::string& key) {
    if (j.count(key)) {
        opt = j[key].get<std::string>();
    }
}

// --- special case for optional values.
template <typename T>
static void update_option(util::optional<T>& opt, const nlohmann::json& j, const std::string& key) {
    if (j.count(key)) {
        auto value = j[key];
        if (value.is_null()) {
            opt = util::nullopt;
        }
        else {
            opt = value.get<T>();
        }
    }
}

// Read options from (optional) json file and command line arguments.
cl_options read_options(int argc, char** argv, bool allow_write) {
    cl_options options;
    std::string save_file = "";

    // Parse command line arguments.
    try {
        cl_options defopts;

        CustomCmdLine cmd("arbor miniapp harness", "0.1");

        TCLAP::ValueArg<std::string> ifile_arg(
            "i", "ifile",
            "read parameters from json-formatted file <file name>",
            false, "","file name", cmd);
        TCLAP::ValueArg<std::string> ofile_arg(
            "o", "ofile",
            "save parameters to json-formatted file <file name>",
            false, "","file name", cmd);
        TCLAP::ValueArg<uint32_t> ncells_arg(
            "n", "ncells", "total number of cells in the model",
            false, defopts.cells, "integer", cmd);
        TCLAP::ValueArg<uint32_t> nsynapses_arg(
            "s", "nsynapses", "number of synapses per cell",
            false, defopts.synapses_per_cell, "integer", cmd);
        TCLAP::ValueArg<std::string> syntype_arg(
            "S", "syntype", "specify synapse type: expsyn or exp2syn",
            false, defopts.syn_type, "string", cmd);
        TCLAP::ValueArg<uint32_t> ncompartments_arg(
            "c", "ncompartments", "number of compartments per segment",
            false, defopts.compartments_per_segment, "integer", cmd);
        TCLAP::ValueArg<double> tfinal_arg(
            "t", "tfinal", "run simulation to <time> ms",
            false, defopts.tfinal, "time", cmd);
        TCLAP::ValueArg<double> dt_arg(
            "d", "dt", "set simulation time step to <time> ms",
            false, defopts.dt, "time", cmd);
        TCLAP::ValueArg<double> bin_dt_arg(
            "", "bin-dt", "set event binning interval to <time> ms",
            false, defopts.bin_dt, "time", cmd);
        TCLAP::SwitchArg bin_regular_arg(
            "","bin-regular","use 'regular' binning policy instead of 'following'", cmd, false);
        TCLAP::SwitchArg all_to_all_arg(
            "m","alltoall","all to all network", cmd, false);
        TCLAP::SwitchArg ring_arg(
            "r","ring","ring network", cmd, false);
        TCLAP::ValueArg<double> sample_dt_arg(
            "", "sample-dt", "set sampling interval to <time> ms",
            false, defopts.bin_dt, "time", cmd);
        TCLAP::ValueArg<double> probe_ratio_arg(
            "p", "probe-ratio", "proportion between 0 and 1 of cells to probe",
            false, defopts.probe_ratio, "proportion", cmd);
        TCLAP::SwitchArg probe_soma_only_arg(
            "X", "probe-soma-only", "only probe cell somas, not dendrites", cmd, false);
        TCLAP::ValueArg<std::string> trace_prefix_arg(
            "P", "prefix", "write traces to files with prefix <prefix>",
            false, defopts.trace_prefix, "string", cmd);
        TCLAP::ValueArg<util::optional<unsigned>> trace_max_gid_arg(
            "T", "trace-max-gid", "only trace probes on cells up to and including <gid>",
            false, defopts.trace_max_gid, "gid", cmd);
        TCLAP::ValueArg<std::string> trace_format_arg(
            "F", "trace-format", "select trace data format: csv or json",
            false, defopts.trace_prefix, "string", cmd);
        TCLAP::ValueArg<util::optional<std::string>> morphologies_arg(
            "M", "morphologies", "load morphologies from SWC files matching <glob>",
            false, defopts.morphologies, "glob", cmd);
        TCLAP::SwitchArg morph_rr_arg(
             "", "morph-rr", "Serial rather than random morphology selection from pool", cmd, false);
        TCLAP::SwitchArg report_compartments_arg(
             "", "report-compartments", "Count compartments in cells before simulation", cmd, false);
        TCLAP::SwitchArg spike_output_arg(
            "f","spike-file-output","save spikes to file", cmd, false);
        TCLAP::ValueArg<unsigned> dry_run_ranks_arg(
            "D","dry-run-ranks","number of ranks in dry run mode",
            false, defopts.dry_run_ranks, "positive integer", cmd);
        TCLAP::SwitchArg profile_only_zero_arg(
             "z", "profile-only-zero", "Only output profile information for rank 0", cmd, false);
        TCLAP::SwitchArg verbose_arg(
             "v", "verbose", "Present more verbose information to stdout", cmd, false);
        TCLAP::ValueArg<std::string> ispike_arg(
            "I", "ispike_file",
            "Input spikes from file",
            false, "", "file name", cmd);


        cmd.reorder_arguments();
        cmd.parse(argc, argv);

        // Handle verbosity separately from other options: it is not considered part
        // of the saved option state.
        options.verbose = verbose_arg.getValue();

        std::string ifile_name = ifile_arg.getValue();
        if (ifile_name != "") {
            // Read parameters from specified JSON file first, to allow
            // overriding arguments on the command line.
            std::ifstream fid(ifile_name);
            if (fid) {
                try {
                    nlohmann::json fopts;
                    fid >> fopts;

                    update_option(options.cells, fopts, "cells");
                    update_option(options.synapses_per_cell, fopts, "synapses");
                    update_option(options.syn_type, fopts, "syn_type");
                    update_option(options.compartments_per_segment, fopts, "compartments");
                    update_option(options.dt, fopts, "dt");
                    update_option(options.bin_dt, fopts, "bin_dt");
                    update_option(options.bin_regular, fopts, "bin_regular");
                    update_option(options.tfinal, fopts, "tfinal");
                    update_option(options.all_to_all, fopts, "all_to_all");
                    update_option(options.ring, fopts, "ring");
                    update_option(options.sample_dt, fopts, "sample_dt");
                    update_option(options.probe_ratio, fopts, "probe_ratio");
                    update_option(options.probe_soma_only, fopts, "probe_soma_only");
                    update_option(options.trace_prefix, fopts, "trace_prefix");
                    update_option(options.trace_max_gid, fopts, "trace_max_gid");
                    update_option(options.trace_format, fopts, "trace_format");
                    update_option(options.morphologies, fopts, "morphologies");
                    update_option(options.morph_rr, fopts, "morph_rr");
                    update_option(options.report_compartments, fopts, "report_compartments");

                    // Parameters for spike output
                    update_option(options.spike_file_output, fopts, "spike_file_output");
                    if (options.spike_file_output) {
                        update_option(options.single_file_per_rank, fopts, "single_file_per_rank");
                        update_option(options.over_write, fopts, "over_write");
                        update_option(options.output_path, fopts, "output_path");
                        update_option(options.file_name, fopts, "file_name");
                        update_option(options.file_extension, fopts, "file_extension");
                    }

                    update_option(options.spike_file_input, fopts, "spike_file_input");
                    if (options.spike_file_input) {
                        update_option(options.input_spike_path, fopts, "input_spike_path");
                    }

                    update_option(options.dry_run_ranks, fopts, "dry_run_ranks");

                    update_option(options.profile_only_zero, fopts, "profile_only_zero");

                }
                catch (std::exception& e) {
                    throw model_description_error(
                        "unable to parse parameters in "+ifile_name+": "+e.what());
                }
            }
            else {
                throw usage_error("unable to open model parameter file "+ifile_name);
            }
        }

        update_option(options.cells, ncells_arg);
        update_option(options.synapses_per_cell, nsynapses_arg);
        update_option(options.syn_type, syntype_arg);
        update_option(options.compartments_per_segment, ncompartments_arg);
        update_option(options.tfinal, tfinal_arg);
        update_option(options.dt, dt_arg);
        update_option(options.bin_dt, bin_dt_arg);
        update_option(options.bin_regular, bin_regular_arg);
        update_option(options.all_to_all, all_to_all_arg);
        update_option(options.ring, ring_arg);
        update_option(options.sample_dt, sample_dt_arg);
        update_option(options.probe_ratio, probe_ratio_arg);
        update_option(options.probe_soma_only, probe_soma_only_arg);
        update_option(options.trace_prefix, trace_prefix_arg);
        update_option(options.trace_max_gid, trace_max_gid_arg);
        update_option(options.trace_format, trace_format_arg);
        update_option(options.morphologies, morphologies_arg);
        update_option(options.morph_rr, morph_rr_arg);
        update_option(options.report_compartments, report_compartments_arg);
        update_option(options.spike_file_output, spike_output_arg);
        update_option(options.profile_only_zero, profile_only_zero_arg);
        update_option(options.dry_run_ranks, dry_run_ranks_arg);

        std::string is_file_name = ispike_arg.getValue();
        if (is_file_name != "") {
            options.spike_file_input = true;
            update_option(options.input_spike_path, ispike_arg);
        }

        if (options.trace_format!="csv" && options.trace_format!="json") {
            throw usage_error("trace format must be one of: csv, json");
        }

        if (options.all_to_all && options.ring) {
            throw usage_error("can specify at most one of --ring and --all-to-all");
        }

        save_file = ofile_arg.getValue();
    }
    catch (TCLAP::ArgException& e) {
        throw usage_error("error parsing command line argument "+e.argId()+": "+e.error());
    }

    // Save option values if requested.
    if (save_file != "" && allow_write) {
        std::ofstream fid(save_file);
        if (fid) {
            try {
                nlohmann::json fopts;

                fopts["cells"] = options.cells;
                fopts["synapses"] = options.synapses_per_cell;
                fopts["syn_type"] = options.syn_type;
                fopts["compartments"] = options.compartments_per_segment;
                fopts["dt"] = options.dt;
                fopts["bin_dt"] = options.bin_dt;
                fopts["bin_regular"] = options.bin_regular;
                fopts["tfinal"] = options.tfinal;
                fopts["all_to_all"] = options.all_to_all;
                fopts["ring"] = options.ring;
                fopts["sample_dt"] = options.sample_dt;
                fopts["probe_ratio"] = options.probe_ratio;
                fopts["probe_soma_only"] = options.probe_soma_only;
                fopts["trace_prefix"] = options.trace_prefix;
                if (options.trace_max_gid) {
                    fopts["trace_max_gid"] = options.trace_max_gid.value();
                }
                else {
                    fopts["trace_max_gid"] = nullptr;
                }
                fopts["trace_format"] = options.trace_format;
                if (options.morphologies) {
                    fopts["morphologies"] = options.morphologies.value();
                }
                else {
                    fopts["morphologies"] = nullptr;
                }
                fopts["morph_rr"] = options.morph_rr;
                fopts["report_compartments"] = options.report_compartments;
                fid << std::setw(3) << fopts << "\n";

            }
            catch (std::exception& e) {
                throw model_description_error(
                    "unable to save parameters in "+save_file+": "+e.what());
            }
        }
        else {
            throw usage_error("unable to write to model parameter file "+save_file);
        }
    }

    // If verbose output requested, emit option summary.
    if (options.verbose) {
        std::cout << options << "\n";
    }

    return options;
}

std::ostream& operator<<(std::ostream& o, const cl_options& options) {
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

} // namespace io
} // namespace arb
