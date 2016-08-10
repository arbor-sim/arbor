#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <numeric>


#include <common_types.hpp>
#include <fvm_cell.hpp>
#include <cell.hpp>
#include <cell_group.hpp>

#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <communication/export_manager.hpp>



using namespace nest::mc;

using global_policy = communication::global_policy;
using lowered_cell = nest::mc::fvm::fvm_cell<double, cell_local_size_type>;
using cell_group_type = cell_group<lowered_cell>;
using time_type = typename cell_group_type::time_type;
using spike_type = communication::exporter_spike_file<time_type,
    global_policy>::spike_type;

int main(int argc, char** argv) {
    // Setup the possible mpi environment
    nest::mc::communication::global_policy_guard global_guard(argc, argv);

    // very simple command line parsing
    if (argc < 3)
    {
        std::cout << "disk_io <int nrspikes> <int nr_repeats> <file_per_rank (true|false)> [simple_output (false|true)]" 
            << "   Simple performance test runner for the exporter manager"
            << "   It exports nrspikes nr_repeats using the export_manager and will produce"
            << "   the total, mean and std of the time needed to perform the output to disk"
            << "   <file_per_rank> true will produce a single file per mpi rank"
            << "   <simple_output> true will produce a simplyfied comma seperated output for automatic parsing"
            << "    The application can be started with mpi support and will produce output on a single rank";

        std::cout << "    if nrspikes is not a multiple of the nr of mpi rank, floor is take" << std::endl;
        exit(1);
    }
    int nr_spikes = atoi(argv[1]);

    if (nr_spikes == 0)
    {
        std::cout << "disk_io <nrspikes>" << std::endl;
        std::cout << "  nrspikes should be a valid integer higher then zero" << std::endl;
        exit(1);
    }
    int nr_repeats = atoi(argv[2]);

    if (nr_repeats == 0)
    {
        std::cout << "disk_io <nrspikes>" << std::endl;
        std::cout << "  nr_repeats should be a valid integer higher then zero" << std::endl;
        exit(1);
    }
    
    bool file_per_rank = false;
    std::string single(argv[3]);
    if (single == std::string("true"))
    {
        file_per_rank = true;
    }

    bool simple_stats = false;
    if (argc == 5)
    {
        std::string simple(argv[4]);
        if (simple == std::string("true")) 
        {
            simple_stats = true;
        }
    }

    // Create the sut  
    nest::mc::communication::export_manager<time_type, global_policy> manager(file_per_rank);

    // We need the nr of ranks to calculate the nr of spikes to produce per
    // rank
    global_policy communication_policy;
    unsigned nr_ranks = communication_policy.size();
    unsigned spikes_per_rank = nr_spikes / nr_ranks;   

    // Create a set of spikes
    std::vector<spike_type> spikes;

    // *********************************************************************
    // To have a realworld data set we get the average spikes per number
    // and use that to get the nr of 'simulated' neurons, and create idxs
    // using this value.
    // Also assume that we have only a single second of simulated time
    // All spike times should be between 0.0 and 1.0:
    unsigned simulated_neurons = spikes_per_rank / 20;
    for (unsigned idx = 0; idx < spikes_per_rank; ++idx) 
    {
        spikes.push_back({ { idx % simulated_neurons, 0 }, 0.0f + 1 / ( 0.05 +idx % 20)});
    }

    std::vector<int> timings;

    int time_total = 0;

    // now output to disk nr_repeats times, while keeping track of the times
    for (int idx = 0; idx < nr_repeats; ++idx) 
    {
        int time_start = clock();

        manager.do_export_rank(spikes);


        int time_stop = clock();
        int run_time = (time_stop - time_start);
        time_total += run_time;
        timings.push_back(run_time);
    }
    

    // Autoput the individual timings to a comma seperated file
    std::ofstream file("file_io_results.csv");
    file << *timings.begin() / double(CLOCKS_PER_SEC) * 1000;
    for (auto time_entry = timings.begin()++; time_entry < timings.end(); ++time_entry) 
    {
        file << "," << *time_entry / double(CLOCKS_PER_SEC) * 1000 ;
    }
    file << std::endl;

    // Calculate some statistics
    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / timings.size();

    std::vector<double> diff(timings.size());
    std::transform(timings.begin(), timings.end(), diff.begin(),
        std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / timings.size());

    if (communication_policy.id() != 0)
    {
        return 0;
    }

    // and output
    if (simple_stats)
    {
        std::cout << time_total / double(CLOCKS_PER_SEC) * 1000 << "," << 
                     mean / double(CLOCKS_PER_SEC) * 1000 << "," << 
                     stdev / double(CLOCKS_PER_SEC) * 1000;
    }
    else
    {
        std::cout << "total time (ms): " << time_total / double(CLOCKS_PER_SEC) * 1000 << std::endl;
        std::cout << "mean  time (ms): " << mean / double(CLOCKS_PER_SEC) * 1000 << std::endl;
        std::cout << "stdev  time (ms): " << stdev / double(CLOCKS_PER_SEC) * 1000 << std::endl;
    }

    return 0;
}
