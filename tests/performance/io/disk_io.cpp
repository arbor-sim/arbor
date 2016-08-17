#include <stdio.h>

#include <fstream>
#include <iostream>
#include <numeric>

#include <cell.hpp>
#include <cell_group.hpp>
#include <common_types.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <fvm_cell.hpp>
#include <io/exporter_spike_file.hpp>
#include <profiling/profiler.hpp>

using namespace nest::mc;

using global_policy = communication::global_policy;
using lowered_cell = fvm::fvm_cell<double, cell_local_size_type>;
using cell_group_type = cell_group<lowered_cell>;
using time_type = typename cell_group_type::time_type;
using spike_type = io::exporter_spike_file<time_type, global_policy>::spike_type;
using timer = util::timer_type;

int main(int argc, char** argv) {

    //Setup the possible mpi environment
    communication::global_policy_guard global_guard(argc, argv);

    // very simple command line parsing
    if (argc < 3) {
        std::cout << "disk_io <int nrspikes> <int nr_repeats>  [simple_output (false|true)]\n"
                  << "   Simple performance test runner for the exporter manager\n"
                  << "   It exports nrspikes nr_repeats using the export_manager and will produce\n"
                  << "   the total, mean and std of the time needed to perform the output to disk\n\n"

                  << "   <file_per_rank> true will produce a single file per mpi rank\n"
                  << "   <simple_output> true will produce a simplyfied comma seperated output for automatic parsing\n\n"

                  << "    The application can be started with mpi support and will produce output on a single rank\n"
                  << "    if nrspikes is not a multiple of the nr of mpi rank, floor is take\n" ;
        return 1;
    }
    auto nr_spikes = atoi(argv[1]);

    if (nr_spikes == 0) {
        std::cout << "disk_io <nrspikes>\n";
        std::cout << "  nrspikes should be a valid integer higher then zero\n";

        return 1;
    }
    auto nr_repeats = atoi(argv[2]);

    if (nr_repeats == 0) {
        std::cout << "disk_io <nrspikes>\n";
        std::cout << "  nr_repeats should be a valid integer higher then zero\n";
        return 1;
    }

    auto simple_stats = false;
    if (argc == 4) {
        std::string simple(argv[3]);
        if (simple == std::string("true"))
        {
            simple_stats = true;
        }
    }

    // Create the sut
    io::exporter_spike_file<time_type, global_policy> exporter(
         "spikes", "./", "gdf", true);

    // We need the nr of ranks to calculate the nr of spikes to produce per
    // rank
    global_policy communication_policy;

    auto nr_ranks = unsigned( communication_policy.size() );
    auto spikes_per_rank = nr_spikes / nr_ranks;

    // Create a set of spikes
    std::vector<spike_type> spikes;

    // *********************************************************************
    // To have a  somewhat realworld data set we calculate from the nr of spikes
    // (assuming 20 hz average) the number of nr of 'simulated' neurons,
    // and create idxs using this value. The number of chars in the number
    // influences the size of the output and thus the speed
    // Also taken that we have only a single second of simulated time
    // all spike times should be between 0.0 and 1.0:
    auto simulated_neurons = spikes_per_rank / 20;
    for (auto idx = unsigned{ 0 }; idx < spikes_per_rank; ++idx) {

        spikes.push_back({
            {idx % simulated_neurons, 0 },   // correct idx
            0.0f + 1 / (0.05f + idx % 20)
        });  // semi random float
    }

    double timings_arr[nr_repeats];
    double time_total = 0;

    // now output to disk nr_repeats times, while keeping track of the times
    for (auto idx = 0; idx < nr_repeats; ++idx) {
        auto time_start = timer::tic();
        exporter.output(spikes);
        auto run_time = timer::toc(time_start);

        time_total += run_time;
        timings_arr[idx] = run_time;
    }

    // create the vector here to prevent changes on the heap influencing the
    // timeing
    std::vector<double> timings;
    for (auto idx = 0; idx < nr_repeats; ++idx) {
        timings.push_back(timings_arr[idx]);
    }


    // Calculate some statistics
    auto sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    auto mean = sum / timings.size();

    std::vector<double> diff(timings.size());
    std::transform(
        timings.begin(), timings.end(), diff.begin(),
        std::bind2nd(std::minus<double>(), mean)
    );
    auto sq_sum = std::inner_product(
        diff.begin(), diff.end(), diff.begin(),
        0.0
    );
    auto stdev = std::sqrt(sq_sum / timings.size());

    auto min = *std::min_element(timings.begin(), timings.end());
    auto max = *std::max_element(timings.begin(), timings.end());


    if (communication_policy.id() != 0) {
        return 0;
    }

    // and output
    if (simple_stats) {
        std::cout << time_total<< ","
                  << mean  << ","
                  << stdev << ","
                  << min << ","
                  << max << std::endl;
    }
    else {
        std::cout << "total time (ms): " << time_total  <<  std::endl;
        std::cout << "mean  time (ms): " << mean <<  std::endl;
        std::cout << "stdev  time (ms): " <<  std::endl;
        std::cout << "min  time (ms): " << min << std::endl;
        std::cout << "max  time (ms): " << max << std::endl;
    }

    return 0;
}
