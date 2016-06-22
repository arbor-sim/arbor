#include <iostream>

#include <cell.hpp>
#include <cell_group.hpp>
#include <fvm_cell.hpp>
#include <mechanism_interface.hpp>

#include "io.hpp"
#include "threading/threading.hpp"
#include "profiling/profiler.hpp"
#include "communication/communicator.hpp"
#include "communication/serial_global_policy.hpp"

using namespace nest;

using real_type = double;
using index_type = int;
using numeric_cell = mc::fvm::fvm_cell<real_type, index_type>;
using cell_group   = mc::cell_group<numeric_cell>;
using communicator_type =
    mc::communication::communicator<mc::communication::serial_global_policy>;

// define some global model parameters
namespace parameters {
namespace synapses {
    // synapse delay
    constexpr double delay  = 5.0;  // ms

    // connection weight
    constexpr double weight = 0.05;  // uS
}
}

///////////////////////////////////////
// prototypes
///////////////////////////////////////

/// make a single abstract cell
mc::cell make_cell(int compartments_per_segment);

/// do basic setup (initialize global state, print banner, etc)
void setup();

/// helper function for initializing cells
cell_group make_lowered_cell(int cell_index, const mc::cell& c);

///////////////////////////////////////
// main
///////////////////////////////////////
int main(void) {

    setup();

    // read parameters
    mc::io::options opt;
    try {
        opt = mc::io::read_options("");
        std::cout << opt << "\n";
    }
    catch (std::exception e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    /////////////////////////////////////////////////////
    //  make cells
    /////////////////////////////////////////////////////

    // make a basic cell
    auto basic_cell = make_cell(opt.compartments_per_segment);

    // make a vector for storing all of the cells
    auto start_init = mc::util::timer_type::tic();
    std::vector<cell_group> cell_groups(opt.cells);

    // initialize the cells in parallel
    mc::threading::parallel_for::apply(
        0, opt.cells,
        [&](int i) {
            // initialize cell
            cell_groups[i] = make_lowered_cell(i, basic_cell);
        }
    );
    auto time_init = mc::util::timer_type::toc(start_init);

    /////////////////////////////////////////////////////
    //  network creation
    /////////////////////////////////////////////////////

    // calculate the source and synapse distribution serially
    auto start_network = mc::util::timer_type::tic();
    std::vector<uint32_t> target_counts(opt.cells);
    std::vector<uint32_t> source_counts(opt.cells);
    for (auto i=0; i<opt.cells; ++i) {
        target_counts[i] = cell_groups[i].cell().synapses()->size();
        source_counts[i] = cell_groups[i].spike_sources().size();
    }

    auto target_map = mc::algorithms::make_index(target_counts);
    auto source_map = mc::algorithms::make_index(source_counts);

    //  create connections
    communicator_type communicator(opt.cells, target_counts);
    for(auto i=0u; i<(uint32_t)opt.cells; ++i) {
        communicator.add_connection({
            i, (i+1)%opt.cells,
            parameters::synapses::weight, parameters::synapses::delay
        });
    }
    communicator.construct();

    auto global_source_map =
        communicator.communication_policy().make_map(source_map.back());
    auto domain_idx = communicator.communication_policy().id();
    for(auto i=0u; i<(uint32_t)opt.cells; ++i) {
        cell_groups[i].set_source_gids(source_map[i]+global_source_map[domain_idx]);
        cell_groups[i].set_target_lids(target_map[i]);
    }

    auto time_network = mc::util::timer_type::toc(start_network);

    /////////////////////////////////////////////////////
    //  time stepping
    /////////////////////////////////////////////////////
    auto start_simulation = mc::util::timer_type::tic();

    auto tfinal = 20.;
    auto t =  0.;
    auto dt = 0.01;
    auto delta = communicator.min_delay();

    communicator.add_spike({opt.cells-1u, 5});

    while(t<tfinal) {
        mc::threading::parallel_for::apply(
            0, opt.cells,
            [&](int i) {
                /*if(communicator.queue(i).size()) {
                    std::cout << ":: delivering events to group " << i << "\n";
                    std::cout << "  " << communicator.queue(i) << "\n";
                }*/
                cell_groups[i].enqueue_events(communicator.queue(i));
                cell_groups[i].advance(t+delta, dt);
                communicator.add_spikes(cell_groups[i].spikes());
                cell_groups[i].clear_spikes();
            }
        );

        communicator.exchange();

        t += delta;
    }

    for(auto i=0u; i<cell_groups.size(); ++i) {
        cell_groups[i].splat("cell"+std::to_string(i)+".txt");
    }

    auto time_simulation = mc::util::timer_type::toc(start_simulation);

    std::cout << "initialization took " << time_init << " s\n";
    std::cout << "network        took " << time_network << " s\n";
    std::cout << "simulation     took " << time_simulation << " s\n";
    std::cout << "performed " << int(tfinal/dt) << " time steps\n";
}

///////////////////////////////////////
// function definitions
///////////////////////////////////////

void setup()
{
    // print banner
    std::cout << "====================\n";
    std::cout << "  starting miniapp\n";
    std::cout << "  - " << mc::threading::description() << " threading support\n";
    std::cout << "====================\n";

    // setup global state for the mechanisms
    mc::mechanisms::setup_mechanism_helpers();
}

// make a high level cell description for use in simulation
mc::cell make_cell(int compartments_per_segment)
{
    nest::mc::cell cell;

    // Soma with diameter 12.6157 um and HH channel
    auto soma = cell.add_soma(12.6157/2.0);
    soma->add_mechanism(mc::hh_parameters());

    // add dendrite of length 200 um and diameter 1 um with passive channel
    std::vector<mc::cable_segment*> dendrites;
    dendrites.push_back(cell.add_cable(0, mc::segmentKind::dendrite, 0.5, 0.5, 200));
    //dendrites.push_back(cell.add_cable(1, mc::segmentKind::dendrite, 0.5, 0.25, 100));
    //dendrites.push_back(cell.add_cable(1, mc::segmentKind::dendrite, 0.5, 0.25, 100));

    for(auto d : dendrites) {
        d->add_mechanism(mc::pas_parameters());
        d->set_compartments(compartments_per_segment);
        d->mechanism("membrane").set("r_L", 100);
    }

    // add stimulus
    //cell.add_stimulus({1,1}, {5., 80., 0.3});

    cell.add_detector({0,0}, 30);
    cell.add_synapse({1, 0.5});

    return cell;
}

cell_group make_lowered_cell(int cell_index, const mc::cell& c)
{
    return cell_group(c);
}

