/*
 * A miniapp that demonstrates using an external spike source.
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>
#include <arborenv/with_mpi.hpp>
#include <mpi.h>

#include "mpiutil.hpp"
#include "parameters-nest.hpp"

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;

//
//  N ranks = Nn + Na
//      Nn = number of nest ranks
//      Na = number of arbor ranks
//
//  Nest  on COMM_WORLD [0, Nn)
//  Arbor on COMM_WORLD [Nn, N)
//

int main(int argc, char** argv) {
    try {
        arborenv::with_mpi guard(argc, argv, false);

        //  INITIALISE MPI

        auto info = get_comm_info(false);

        //  MODEL SETUP
        auto params = read_options_nest(argc, argv);

        int num_nest_cells = params.num_cells;
        float nest_min_delay = params.min_delay;


        //  HAND SHAKE ARBOR-NEST
        on_local_rank_zero(info, [&] {
                std::cout << "NEST: starting handshake" << std::endl;
        });

        // hand shake #1: communicate cell populations
        int num_arbor_cells = broadcast(0,   MPI_COMM_WORLD, info.arbor_root);
        broadcast(num_nest_cells,   MPI_COMM_WORLD, info.nest_root);
        int  total_cells = num_arbor_cells + num_nest_cells;

        on_local_rank_zero(info, [&] {
                std::cout << "NEST: num_nest_cells: " << num_nest_cells << ", "
                          << "num_arbor_cells: " << num_arbor_cells << ", "
                          << "total_cells: " << total_cells
                          << std::endl;
        });

        // hand shake #2: min delay
        float arb_comm_time = broadcast(0.f, MPI_COMM_WORLD, info.arbor_root);
        float nest_comm_time = nest_min_delay;
        broadcast(nest_comm_time, MPI_COMM_WORLD, info.nest_root);
        float min_delay = std::min(nest_comm_time, arb_comm_time);
        
        on_local_rank_zero(info, [&] {
                std::cout << "NEST: min_delay=" << min_delay << std::endl;
        });


        float delta = min_delay;
        float sim_duration = params.duration;
        unsigned steps = sim_duration/delta;
        if (steps*delta<sim_duration) ++steps;

        // hand shake #3: steps
        unsigned steps_arbor = broadcast(0u, MPI_COMM_WORLD, info.arbor_root);

        on_local_rank_zero(info, [&] {
                std::cout << "NEST: delta=" << delta << ", "
                          << "sim_duration=" << sim_duration << ", "
                          << "steps=" << steps
                          << std::endl;
        });

        //  BUILD NEST PROXY MODEL
        std::vector<int> local_cells;
        for (int gid=num_arbor_cells+info.local_rank; gid<total_cells; gid+=info.nest_size) {
            local_cells.push_back(gid);
        }
        print_vec_comm("NEST", local_cells, info.comm);

        //  SEND SPIKES TO ARBOR (RUN SIMULATION)
        for (unsigned step=0; step<=steps; ++step) {
            if (step > steps_arbor) {
                throw std::runtime_error(std::string("Bad step: ") + std::to_string(step) + " > " + std::to_string(steps_arbor));
            }
                
            on_local_rank_zero(info, [&] {
                    std::cout << "NEST: callback " << step << " at t " << step*delta << std::endl;
            });

            std::vector<arb::spike> local_spikes;
            if (!step) {
                for (unsigned gid: local_cells) {
                    arb::spike s;
                    s.source = {gid, 0u};
                    s.time = (float)(gid-num_arbor_cells);
                    local_spikes.push_back(s);
                }
            }
            print_vec_comm("NEST-send", local_spikes, info.comm);
            static int stepn = 0;
            std::cerr << "NEST: step " << stepn++ << std::endl;
            auto v = gather_spikes(local_spikes, MPI_COMM_WORLD);
            if (v.size()) print_vec_comm("NEST-recv", v, info.comm);
        }

        if (steps != steps_arbor) {
            throw std::runtime_error(std::string("Bad step: ") + std::to_string(steps) + " < " + std::to_string(steps_arbor));
        }
        
        on_local_rank_zero(info, [&] {
                std::cout << "NEST: reached end" << std::endl;
        });
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in nest proxy:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}
