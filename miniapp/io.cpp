#include <exception>
#include <fstream>
#include <istream>

#include <tclap/CmdLine.h>
#include <json/src/json.hpp>

#include <util/optional.hpp>

#include "io.hpp"

// Let TCLAP understand value arguments that are of an optional type.

namespace TCLAP {

template <typename V>
struct ArgTraits<nest::mc::util::optional<V>> {
    using ValueCategory = ValueLike;
};

} // namespace TCLAP

namespace nest {
namespace mc {

// Using static here because we do not want external linkage for this operator

namespace util {
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

/// read simulation options from json file with name fname
/// if file name is empty or if file is not a valid json file a default
/// set of parameters is returned :
///      1000 cells, 500 synapses per cell, 100 compartments per segment
cl_options read_options(int argc, char** argv) {

    // set default options
    const cl_options defopts{"", 1000, 500, "expsyn", 100, 100., 0.025, false,
                             false, 1.0, "trace_", util::nothing};

    cl_options options;
    // parse command line arguments
    try {
        TCLAP::CmdLine cmd("mod2c performance benchmark harness", ' ', "0.1");

        TCLAP::ValueArg<uint32_t> ncells_arg(
            "n", "ncells", "total number of cells in the model",
            false, defopts.cells, "non negative integer", cmd);
        TCLAP::ValueArg<uint32_t> nsynapses_arg(
            "s", "nsynapses", "number of synapses per cell",
            false, defopts.synapses_per_cell, "non negative integer", cmd);
        TCLAP::ValueArg<std::string> syntype_arg(
            "S", "syntype", "type of synapse (expsyn or exp2syn)",
            false, defopts.syn_type, "synapse type", cmd);
        TCLAP::ValueArg<uint32_t> ncompartments_arg(
            "c", "ncompartments", "number of compartments per segment",
            false, defopts.compartments_per_segment, "non negative integer", cmd);
        TCLAP::ValueArg<std::string> ifile_arg(
            "i", "ifile", "json file with model parameters",
            false, "","file name string", cmd);
        TCLAP::ValueArg<double> tfinal_arg(
            "t", "tfinal", "time to simulate in ms",
            false, defopts.tfinal, "positive real number", cmd);
        TCLAP::ValueArg<double> dt_arg(
            "d", "dt", "time step size in ms",
            false, defopts.dt, "positive real number", cmd);
        TCLAP::SwitchArg all_to_all_arg(
            "m","alltoall","all to all network", cmd, false);
        TCLAP::ValueArg<double> probe_ratio_arg(
            "p", "probe-ratio", "proportion of cells to probe",
            false, defopts.probe_ratio, "real number in [0,1]", cmd);
        TCLAP::SwitchArg probe_soma_only_arg(
            "X", "probe-soma-only", "only probe cell somas, not dendrites", cmd, false);
        TCLAP::ValueArg<std::string> trace_prefix_arg(
            "P", "trace-prefix", "write traces to files with this prefix",
            false, defopts.trace_prefix, "stringr", cmd);
        TCLAP::ValueArg<util::optional<unsigned>> trace_max_gid_arg(
            "T", "trace-max-gid", "only trace probes on cells up to this gid",
            false, defopts.trace_max_gid, "unisgned integer", cmd);

        cmd.parse(argc, argv);

        options.cells = ncells_arg.getValue();
        options.synapses_per_cell = nsynapses_arg.getValue();
        options.syn_type = syntype_arg.getValue();
        options.compartments_per_segment = ncompartments_arg.getValue();
        options.ifname = ifile_arg.getValue();
        options.tfinal = tfinal_arg.getValue();
        options.dt = dt_arg.getValue();
        options.all_to_all = all_to_all_arg.getValue();
        options.probe_ratio = probe_ratio_arg.getValue();
        options.probe_soma_only = probe_soma_only_arg.getValue();
        options.trace_prefix = trace_prefix_arg.getValue();
        options.trace_max_gid = trace_max_gid_arg.getValue();
    }
    // catch any exceptions in command line handling
    catch (TCLAP::ArgException& e) {
        throw usage_error("error parsing command line argument "+e.argId()+": "+e.error());
    }

    if (options.ifname != "") {
        std::ifstream fid(options.ifname);
        if (fid) {
            // read json data in input file
            nlohmann::json fopts;
            fid >> fopts;

            try {
                options.cells = fopts["cells"];
                options.synapses_per_cell = fopts["synapses"];
                options.compartments_per_segment = fopts["compartments"];
                options.dt = fopts["dt"];
                options.tfinal = fopts["tfinal"];
                options.all_to_all = fopts["all_to_all"];
            }
            catch (std::exception& e) {
                throw model_description_error(
                    "unable to parse parameters in "+options.ifname+": "+e.what());
            }
        }
        else {
            throw usage_error("unable to open model parameter file "+options.ifname);
        }
    }

    return options;
}

std::ostream& operator<<(std::ostream& o, const cl_options& options) {
    o << "simultion options:\n";
    o << "  cells                : " << options.cells << "\n";
    o << "  compartments/segment : " << options.compartments_per_segment << "\n";
    o << "  synapses/cell        : " << options.synapses_per_cell << "\n";
    o << "  simulation time      : " << options.tfinal << "\n";
    o << "  dt                   : " << options.dt << "\n";
    o << "  all to all network   : " << (options.all_to_all ? "yes" : "no") << "\n";
    o << "  probe ratio          : " << options.probe_ratio << "\n";
    o << "  probe soma only      : " << (options.probe_soma_only ? "yes" : "no") << "\n";
    o << "  trace prefix         : " << options.trace_prefix << "\n";
    o << "  trace max gid        : ";
    if (options.trace_max_gid) {
       o << *options.trace_max_gid;
    }
    o << "\n";
    o << "  input file name      : " << options.ifname << "\n";

    return o;
}

} // namespace io
} // namespace mc
} // namespace nest
