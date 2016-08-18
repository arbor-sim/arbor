#pragma once

#include <string>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <utility>

#include <util/optional.hpp>

namespace nest {
namespace mc {
namespace io {

// holds the options for a simulation run
struct cl_options {
    uint32_t cells;
    uint32_t synapses_per_cell;
    std::string syn_type;
    uint32_t compartments_per_segment;
    double tfinal;
    double dt;
    bool all_to_all;
    bool probe_soma_only;
    double probe_ratio;
    std::string trace_prefix;
    util::optional<unsigned> trace_max_gid;

    // Parameters for spike output
    bool spike_file_output;
    bool single_file_per_rank;
    bool over_write;
    std::string output_path;
    std::string file_name;
    std::string file_extension;
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

cl_options read_options(int argc, char** argv);


} // namespace io
} // namespace mc
} // namespace nest
