#include <fstream>
#include <vector>
#include <utility>
#include <common_types.hpp>
#include <exception>

#include <tests/simple_recipes.hpp>

#include <ipss_cell_description.hpp>
#include <ipss_cell_group.hpp>

namespace ipss_impl {
    using ipss_recipe = arb::homogeneous_recipe<arb::cell_kind::inhomogeneous_poisson_spike_source,
        arb::ipss_cell_description>;

    // Simple exception class that can be instantiated with a string as what msg
    class ipss_error : public std::exception
    {
    public:
        ipss_error(std::string what) : what_(what) {}

    private:
        virtual const char* what() const throw() {
            return what_.c_str();
        }

    private:
        std::string what_;
    };

    // Parses time rate pairs separated with a comma from file returns them
    // as a vector of time, rate pairs
    // Throw and ipss_error when the file could not be opened
    // If errors occur during parsing they are eaten silently and parsing stops
    std::vector<std::pair<arb::time_type, double>> parse_time_rate_from_path(std::string path) {
        std::ifstream infile(path);
        arb::time_type time;
        double rate;
        char comma;

        std::vector<std::pair<arb::time_type, double>> pairs;

        if (infile) {
            while (infile >> time >> comma >> rate) {
                pairs.push_back({ time,rate });
            }
        }
        else {
            throw ipss_error("Could not open supplied time_rate file");
        }

        return pairs;
    }

    // Return our default vector of time rate pairs:
    // 'plot' of the default rates:
    //  |
    //  |     _-_
    //  |    -   -  -
    //  |   -     -- -
    //  |__-__________-__
    std::vector<std::pair<arb::time_type, double>> default_time_rate_pairs(){
        std::vector<std::pair<arb::time_type, double>> pairs;

        double mult = 30.0;
        pairs.push_back({ 0.0, 0.0 * mult});
        pairs.push_back({ 50.0, 0.0 * mult });
        pairs.push_back({ 100.0, 1.0 * mult });
        pairs.push_back({ 200.0, 5.0 * mult });
        pairs.push_back({ 300.0, 7.0 * mult });
        pairs.push_back({ 400.0, 8.0 * mult });
        pairs.push_back({ 500.0, 7.0 * mult });
        pairs.push_back({ 600.0, 3.0 * mult });
        pairs.push_back({ 700.0, 3.0 * mult });
        pairs.push_back({ 750.0, 5.0 * mult });
        pairs.push_back({ 800.0, 2.5 * mult });
        pairs.push_back({ 900.0, 0.0 * mult });
        pairs.push_back({ 1000.0, 0.0 * mult });

        return pairs;
    }

    // Creates a ipss cell group with the supplied time rate vector
    // run the for the supplied time range (in ten steps)
    // collect the spikes and return these
    std::vector<arb::spike> create_and_run_ipss_cell_group(arb::cell_gid_type n_cells,
        arb::time_type begin, arb::time_type end, arb::time_type sample_delta,
        std::vector<std::pair<arb::time_type, double>> rates_per_time, bool interpolate) {
        // Create the cell indexes
        std::vector<arb::cell_gid_type> gids;
        for (arb::cell_gid_type idx = 0; idx < n_cells; ++idx) {
            gids.push_back(idx);
        }

        // create the cell group
        arb::ipss_cell_group sut(gids,
            ipss_recipe(n_cells,
            { begin, end, sample_delta, rates_per_time, interpolate }));

        // Split whatever time we have into 10 equal epochs
        arb::time_type time_step = (end - begin) / 10.0;

        // run the cell
        for (std::size_t idx = 0; idx < 10; ++idx) {
            sut.advance({ idx, begin + (idx + 1) * time_step }, 0.01, {});
        }

        // return a copy, the cells will be destructed after the end
        // of this function
        return std::vector<arb::spike>(sut.spikes());
    }

    // Simple spike writer output spikes according to the gpf standard
    // gid, time
    void write_spikes_to_path(const std::vector<arb::spike> spikes, std::string path) {
        std::ofstream outfile(path);
        if (outfile) {
            for (auto spike : spikes) {
                outfile << spike.source.gid << "," << spike.time << "\n";
            }
        }
        else {
            throw ipss_error("Could not open supplied output path for writing spikes");
        }
    }
}
