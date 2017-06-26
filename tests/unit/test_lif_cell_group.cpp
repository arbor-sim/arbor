#include "../gtest.h"

#include <common_types.hpp>
#include <lif_cell_group.hpp>
#include <util/rangeutil.hpp>
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
    events.push_back({{0, 0}, 1, 100});
    
    // this event happens inside the refractory period of the previous
    // event, thus, should be ignored (first neuron)
    events.push_back({{0, 0}, 1.1, 100});
    
    // this event happens long after the refractory period of the previous
    // event, should thus trigger new spike (first neuron)
    events.push_back({{0, 0}, 50, 100});
    
    // this is event to the second neuron
    events.push_back({{1, 0}, 1, 100});

    
    group->enqueue_events(events);
    group->advance(100, 0.01);
    std::vector<spike> spikes = group->spikes();
    
    EXPECT_EQ(3, spikes.size());
    
    
    
}
