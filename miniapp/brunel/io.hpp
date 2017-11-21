#pragma once

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
        uint32_t nexc = 400;
        uint32_t ninh = 100;
        uint32_t next = 40;
        double syn_per_cell_prop = 0.05;
        float weight = 1.2;
        float delay = 0.1;
        float rel_inh_strength = 1;
        double poiss_rate = 1;

        // Simulation running parameters:
        double tfinal = 100.;
        double dt = 1;
        uint32_t group_size = 10;

        // Parameters for spike output.
        bool spike_file_output = false;
        bool single_file_per_rank = false;
        bool over_write = true;
        std::string output_path = "./";
        std::string file_name = "spikes";
        std::string file_extension = "gdf";

        // Turn on/off profiling output for all ranks.
        bool profile_only_zero = false;

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
} // namespace io
} // namespace arbor
