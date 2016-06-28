#include <iostream>

#include <cell.hpp>
#include <cell_group.hpp>
#include <fvm_cell.hpp>
#include <mechanism_interface.hpp>

#include "io.hpp"
#include "threading/threading.hpp"
#include "profiling/profiler.hpp"
#include "communication/communicator.hpp"
#include "communication/global_policy.hpp"

using namespace nest;

using real_type = double;
using index_type = int;
using id_type = uint32_t;
using numeric_cell = mc::fvm::fvm_cell<real_type, index_type>;
using cell_group   = mc::cell_group<numeric_cell>;

using global_policy = nest::mc::communication::global_policy;
using communicator_type =
    mc::communication::communicator<global_policy>;

struct model {
    communicator_type communicator;
    std::vector<cell_group> cell_groups;

    int num_groups() const {
        return cell_groups.size();
    }

    void run(double tfinal, double dt) {
        auto t = 0.;
        auto delta = communicator.min_delay();
        while(t<tfinal) {
            mc::threading::parallel_for::apply(
                0, num_groups(),
                [&](int i) {
                        mc::util::profiler_enter("stepping","events");
                    cell_groups[i].enqueue_events(communicator.queue(i));
                        mc::util::profiler_leave();
                    cell_groups[i].advance(t+delta, dt);
                        mc::util::profiler_enter("events");
                    communicator.add_spikes(cell_groups[i].spikes());
                    cell_groups[i].clear_spikes();
                        mc::util::profiler_leave(2);
                }
            );

                mc::util::profiler_enter("stepping", "exchange");
            communicator.exchange();
                mc::util::profiler_leave(2);

            t += delta;
        }
    }

    void init_communicator() {
            mc::util::profiler_enter("setup", "communicator");

        // calculate the source and synapse distribution serially
        std::vector<id_type> target_counts(num_groups());
        std::vector<id_type> source_counts(num_groups());
        for (auto i=0; i<num_groups(); ++i) {
            target_counts[i] = cell_groups[i].cell().synapses()->size();
            source_counts[i] = cell_groups[i].spike_sources().size();
        }

        target_map = mc::algorithms::make_index(target_counts);
        source_map = mc::algorithms::make_index(source_counts);

        //  create connections
        communicator = communicator_type(num_groups(), target_counts);

            mc::util::profiler_leave(2);
    }

    void update_gids() {
            mc::util::profiler_enter("setup", "globalize");
        auto com_policy = communicator.communication_policy();
        auto global_source_map = com_policy.make_map(source_map.back());
        auto domain_idx = communicator.domain_id();
        for (auto i=0; i<num_groups(); ++i) {
            cell_groups[i].set_source_gids(source_map[i]+global_source_map[domain_idx]);
            cell_groups[i].set_target_gids(target_map[i]+communicator.target_gid_from_group_lid(0));
        }
            mc::util::profiler_leave(2);
    }

    // TODO : only stored here because init_communicator() and update_gids() are split
    std::vector<id_type> source_map;
    std::vector<id_type> target_map;
};

// define some global model parameters
namespace parameters {
namespace synapses {
    // synapse delay
    constexpr double delay  = 5.0;  // ms

    // connection weight
    constexpr double weight = 0.005;  // uS
}
}

///////////////////////////////////////
// prototypes
///////////////////////////////////////

/// make a single abstract cell
mc::cell make_cell(int compartments_per_segment, int num_synapses);

/// do basic setup (initialize global state, print banner, etc)
void setup();

/// helper function for initializing cells
cell_group make_lowered_cell(int cell_index, const mc::cell& c);

/// models
void ring_model(nest::mc::io::options& opt, model& m);
void all_to_all_model(nest::mc::io::options& opt, model& m);


///////////////////////////////////////
// main
///////////////////////////////////////
int main(int argc, char** argv) {
    nest::mc::communication::global_policy_guard global_guard(argc, argv);

    setup();

    // read parameters
    mc::io::options opt;
    try {
        opt = mc::io::read_options("");
        if (!global_policy::id()) {
            std::cout << opt << "\n";
        }
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    model m;
    all_to_all_model(opt, m);

    //
    //  time stepping
    //
    auto tfinal = 200.;
    auto dt = 0.01;

    auto id = m.communicator.domain_id();

    if (!id) {
        m.communicator.add_spike({0, 5});
    }

    m.run(tfinal, dt);

    mc::util::profiler_output(0.001);

    if (!id) {
        std::cout << "there were " << m.communicator.num_spikes() << " spikes\n";
    }

#ifdef SPLAT
    if (!global_policy::id()) {
        //for (auto i=0u; i<m.cell_groups.size(); ++i) {
        m.cell_groups[0].splat("cell0.txt");
        m.cell_groups[1].splat("cell1.txt");
        m.cell_groups[2].splat("cell2.txt");
        //}
    }
#endif
}

///////////////////////////////////////
// models
///////////////////////////////////////

void ring_model(nest::mc::io::options& opt, model& m) {
    //
    //  make cells
    //

    // make a basic cell
    auto basic_cell = make_cell(opt.compartments_per_segment, 1);

    // make a vector for storing all of the cells
    m.cell_groups = std::vector<cell_group>(opt.cells);

    // initialize the cells in parallel
    mc::threading::parallel_for::apply(
        0, opt.cells,
        [&](int i) {
            // initialize cell
                mc::util::profiler_enter("setup");
                    mc::util::profiler_enter("make cell");
            m.cell_groups[i] = make_lowered_cell(i, basic_cell);
                    mc::util::profiler_leave();
                mc::util::profiler_leave();
        }
    );

    //
    //  network creation
    //
    m.init_communicator();

    for (auto i=0u; i<(id_type)opt.cells; ++i) {
        m.communicator.add_connection({
            i, (i+1)%opt.cells,
            parameters::synapses::weight, parameters::synapses::delay
        });
    }

    m.update_gids();
}

void all_to_all_model(nest::mc::io::options& opt, model& m) {
    //
    //  make cells
    //

    // make a basic cell
    auto basic_cell = make_cell(opt.compartments_per_segment, opt.cells-1);

    // make a vector for storing all of the cells
    id_type ncell_global = opt.cells;
    id_type ncell_local  = ncell_global / m.communicator.num_domains();
    int remainder = ncell_global - (ncell_local*m.communicator.num_domains());
    if (m.communicator.domain_id()<remainder) {
        ncell_local++;
    }

    m.cell_groups = std::vector<cell_group>(ncell_local);

    // initialize the cells in parallel
    mc::threading::parallel_for::apply(
        0, ncell_local,
        [&](int i) {
                mc::util::profiler_enter("setup", "cells");
            m.cell_groups[i] = make_lowered_cell(i, basic_cell);
                mc::util::profiler_leave(2);
        }
    );

    //
    //  network creation
    //
    m.init_communicator();

        mc::util::profiler_enter("setup", "connections");
    // lid is local cell/group id
    for (auto lid=0u; lid<ncell_local; ++lid) {
        auto target = m.communicator.target_gid_from_group_lid(lid);
        auto gid = m.communicator.group_gid_from_group_lid(lid);
        // tid is global cell/group id
        for (auto tid=0u; tid<ncell_global; ++tid) {
            if (gid!=tid) {
                m.communicator.add_connection({
                    tid, target++,
                    parameters::synapses::weight, parameters::synapses::delay
                });
            }
        }
    }

    m.communicator.construct();
        mc::util::profiler_leave(2);

    m.update_gids();
}

///////////////////////////////////////
// function definitions
///////////////////////////////////////

void setup() {
    // print banner
    if (!global_policy::id()) {
        std::cout << "====================\n";
        std::cout << "  starting miniapp\n";
        std::cout << "  - " << mc::threading::description() << " threading support\n";
        std::cout << "  - communication policy: " << global_policy::name() << "\n";
        std::cout << "====================\n";
    }

    // setup global state for the mechanisms
    mc::mechanisms::setup_mechanism_helpers();
}

// make a high level cell description for use in simulation
mc::cell make_cell(int compartments_per_segment, int num_synapses) {
    nest::mc::cell cell;

    // Soma with diameter 12.6157 um and HH channel
    auto soma = cell.add_soma(12.6157/2.0);
    soma->add_mechanism(mc::hh_parameters());

    // add dendrite of length 200 um and diameter 1 um with passive channel
    std::vector<mc::cable_segment*> dendrites;
    dendrites.push_back(cell.add_cable(0, mc::segmentKind::dendrite, 0.5, 0.5, 200));
    //dendrites.push_back(cell.add_cable(1, mc::segmentKind::dendrite, 0.5, 0.25, 100));
    //dendrites.push_back(cell.add_cable(1, mc::segmentKind::dendrite, 0.5, 0.25, 100));

    for (auto d : dendrites) {
        d->add_mechanism(mc::pas_parameters());
        d->set_compartments(compartments_per_segment);
        d->mechanism("membrane").set("r_L", 100);
    }

    // add stimulus
    //cell.add_stimulus({1,1}, {5., 80., 0.3});

    cell.add_detector({0,0}, 30);

    for (auto i=0; i<num_synapses; ++i) {
        cell.add_synapse({1, 0.5});
    }

    return cell;
}

cell_group make_lowered_cell(int cell_index, const mc::cell& c) {
    return cell_group(c);
}

