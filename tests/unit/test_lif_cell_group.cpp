#include "../gtest.h"

#include <common_types.hpp>
#include <fstream>
#include <lif_cell_group.hpp>
#include <util/rangeutil.hpp>
#include <util/any.hpp>
#include <recipe.hpp>
#include <lif_cell_description.hpp>
#include "common.hpp"
#include <util/unique_any.hpp>
#include <backends.hpp>
#include <cell_group.hpp>
#include <fvm_multicell.hpp>
#include <mc_cell_group.hpp>
#include <util/unique_any.hpp>
#include <lif_cell_group.hpp>
#include <cell_group_factory.hpp>
#include <event_queue.hpp>

using namespace nest::mc;
using mc_fvm_cell = mc_cell_group<fvm::fvm_multicell<multicore::backend>>;

class ring_recipe: nest::mc::recipe {
    
public:
    
    ring_recipe(cell_size_type n, float weight, float delay):
        ncells_(n), weight_(weight), delay_(delay) {}
    
    cell_size_type num_cells() const override {
        return ncells_;
    }
    
    cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::lif_neuron;
    }
    
    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<cell_connection> connections;
        
        cell_connection conn;
        
        conn.weight = weight_;
        conn.delay = delay_;
        
        // for source - {source_idx, id of the compartment from which spike is coming}
        // for single compartment cells, the second parameter is always 0
        
        // for dest - {dest_idx, lid of synapse coming to dest}
        // since there is only 1 synapse coming into dest
        // this parameter is also 0
        
        conn.source = {(gid + ncells_ - 1) % ncells_, 0};
        conn.dest =   {gid, 0};
        
        connections.push_back(conn);
        
        return connections;
        
    }
    
    util::unique_any get_cell_description(cell_gid_type) const override {
        return lif_cell_description();
    }
    
    cell_count_info get_cell_count_info(cell_gid_type) const override {
        return {1u, 1u, 0u};
    }
    
private:
    int ncells_;
    
    float weight_, delay_;
    
};


TEST(lif_cell_group, recipe)
{
    ring_recipe rr(100, 1, 0.1);
    
    EXPECT_EQ(100, rr.num_cells());
    
    EXPECT_EQ(1, rr.connections_on(0u).size());
    
    EXPECT_EQ(1, rr.connections_on(55u).size());
    
    EXPECT_EQ(0, rr.connections_on(1u)[0].source.gid);
    
    EXPECT_EQ(99, rr.connections_on(0u)[0].source.gid);

}


TEST(lif_cell_group, cell_group_factory) {
    std::vector<util::unique_any> cells;
    cells.emplace_back(lif_cell_description());
    cells.emplace_back(lif_cell_description());
    
    cell_group_ptr group = cell_group_factory(
                                              cell_kind::lif_neuron,
                                              0,
                                              cells,
                                              backend_policy::use_multicore);
    
    std::vector<postsynaptic_spike_event> events;
    
    // first event to trigger the spike (first neuron)
    events.push_back({{0, 0}, 1, 1000});
    
    // this event happens inside the refractory period of the previous
    // event, thus, should be ignored (first neuron)
    events.push_back({{0, 0}, 1.1, 1000});
    
    // this event happens long after the refractory period of the previous
    // event, should thus trigger new spike (first neuron)
    events.push_back({{0, 0}, 50, 1000});
    
    // this is event to the second neuron
    events.push_back({{1, 0}, 1, 1000});

    
    group->enqueue_events(events);
    group->advance(100, 0.01);
    std::vector<spike> spikes = group->spikes();
    
    EXPECT_EQ(3, spikes.size());
}

TEST(lif_cell_group, spikes_testing) {
    std::vector<util::unique_any> cells;
    cells.emplace_back(lif_cell_description());
    
  
    
    std::unique_ptr<lif_cell_group> group = std::unique_ptr<lif_cell_group>(static_cast<lif_cell_group*>(cell_group_factory(
                                                                             cell_kind::lif_neuron,
                                                                             0,
                                                                             cells,
                                                                             backend_policy::use_multicore).release()));
    
    std::vector<postsynaptic_spike_event> events;
    
    std::vector<time_type> incoming_spikes;
    
    time_type simulation_end = 50;
    
    // add events at times i for the first 80% time of the simulation
    for(int i = 1; i < (int) (0.8 * simulation_end); i++) {
        // last parameter is the weight
        events.push_back({{0, 0}, static_cast<time_type>(i), 100});
        
        incoming_spikes.push_back(i);
    }
    
    group->enqueue_events(events);
    
    group->turn_on_sampling(0.01);
    
    // second parameter is dt, but is ignored
    group->advance(simulation_end, 0.01);
    std::vector<spike> spikes = group->spikes();
    
    std::vector<std::pair<time_type, lif_cell_description::value_type> > voltage = group->voltage();
    
    std::ofstream in_spikes_file;
    in_spikes_file.open("../../tests/unit/lif_neuron_input_spikes.txt");
    
    std::ofstream out_spikes_file;
    out_spikes_file.open("../../tests/unit/lif_neuron_output_spikes.txt");
    
    std::ofstream voltage_file;
    voltage_file.open("../../tests/unit/lif_neuron_voltage.txt");
    
    
    for(auto& in_spike : incoming_spikes) {
        in_spikes_file << in_spike << std::endl;
    }
    
    for(auto & out_spike : spikes) {
        out_spikes_file << out_spike.time << std::endl;
    }
    
    for(auto & v : voltage) {
        voltage_file << v.first << " " << v.second << std::endl;
    }
    
    
    in_spikes_file.close();
    out_spikes_file.close();
    voltage_file.close();
    
    
}
