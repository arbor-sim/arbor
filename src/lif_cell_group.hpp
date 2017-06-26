#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

#include <algorithms.hpp>
#include <cell_group.hpp>
#include <lif_cell_description.hpp>
#include <common_types.hpp>
#include <event_binner.hpp>
#include <event_queue.hpp>
#include <recipe.hpp>
#include <sampler_function.hpp>
#include <spike.hpp>
#include <util/debug.hpp>
#include <util/partition.hpp>
#include <util/range.hpp>
#include <util/unique_any.hpp>

#include <profiling/profiler.hpp>

namespace nest {
    namespace mc {
        
class lif_cell_group: public cell_group {
public:
    
    using value_type = lif_cell_description::value_type;

    lif_cell_group() = default;

    // constructor containing gid of first cell in a group and a container of all cells
    // same type as in recipe cells
    
    lif_cell_group(cell_gid_type first_gid, const std::vector<util::unique_any>& cells):gid_base_{first_gid} {
        
        cells_.reserve(cells.size());
        
        // can be done with tranform view (acts like a map, but with lazy evaluation, only at the request)
        for(const auto& x : cells) {
            cells_.push_back(util::any_cast<lif_cell_description>(x));
        }
        
        cell_events_.resize(cells_.size());
        
        last_time_updated_.resize(cells.size(), 0);
    }

    // TODO: implement lif_neuron cell kind in cell_factory
    cell_kind get_cell_kind() const override {
        return cell_kind::lif_neuron;
    }
    
    // advances a single cell (lid) with Euler method
    void advance_cell(time_type tfinal, time_type dt, cell_gid_type lid) {
        
        time_type& tstart = last_time_updated_[lid];
        lif_cell_description& cell = cells_[lid];
        
        // if a neuron was in the refractory period
        // ignore any new events that happened before tstart
        while (cell_events_[lid].pop_if_before(tstart));
            
        // integrate until tfinal in steps of dt = min(t_final, t_event)
        // TODO: add alpha-kernel
        while(auto ev = cell_events_[lid].pop_if_before(tfinal)) {
            // weight has explicit float type in postsynaptic_spike_event
            float weight = ev->weight;
            time_type spike_time = ev->time;
            
            // if a neuron is in refractory period
            // then ignore this spike
            if(spike_time < tstart)
                continue;
            
            /*
              THIS EXACT SOLUTION HOLDS UNDER THE FOLLOWING ASSUMPTIONS:
             
              1) I_e = 0
              2) no alpha kernel
             
             */
            
            // let the membrane potential decay
            cell.V_m *= exp(-(spike_time - tstart) / cell.tau_m);
            // add jump due to spike
            cell.V_m += weight/cell.C_m;
            
            // if spike occured
            if(cell.V_m >= cell.V_th) {
                cell_member_type spike_neuron_gid = {gid_base_ + lid, 0};
                spike s = {spike_neuron_gid, tstart};
                
                spikes_.push_back(s);
                
                // advance last_time_updated
                tstart += cell.t_ref;
                // reset the voltage to resting potential
                cell.V_m = cell.E_L;
                
                // TODO: should reset the voltage to V_reset instead and let it
                // exponentially grow to E_L inside the refractory period
            }
        }
    }
    
    void advance(time_type tfinal, time_type dt) override {
        
        // distribute incoming events to individual cells
        // this can be done efficiently using GPU
        while (auto ev = events_.pop_if_before(tfinal)) {
            int target_gid = ev->target.gid;
            //double spike_time = ev->time;
            //double weight = ev->weight;
            
            // gid -> lid
            // get value_type from optional<value_type> with get!
            cell_events_[target_gid - gid_base_].push(ev.get());
        }
        
        // can be done efficiently with CUDA
        for(int i = 0; i < cells_.size(); ++i) {
            advance_cell(tfinal, dt, i);
        }
    }
    
    void enqueue_events(const std::vector<postsynaptic_spike_event>& events) override {
        for (auto& e: events) {
            events_.push(e);
        }
    }
    
    const std::vector<spike>& spikes() const override {
        return spikes_;
    }
    
    void clear_spikes() override {
        spikes_.clear();
    }
    
    // TODO: implement sampler
    void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) override {}
    
    // TODO: implement binner_
    void set_binning_policy(binning_kind policy, time_type bin_interval) override {
    }
    
    // no probes in single-compartment cells
    std::vector<probe_record> probes() const override {
        return std::vector<probe_record>();
    }
    
    void reset() override {
        spikes_.clear();
        events_.clear();
        cells_.clear();
    }
    
    
private:
    
    // gid of first cell in group.
    cell_gid_type gid_base_;
        
    // cells that belong to this group
    std::vector<lif_cell_description> cells_;
    
    // Spikes that are generated (not necessarily sorted)
    std::vector<spike> spikes_;
    
    // Pending events to be delivered.
    event_queue<postsynaptic_spike_event> events_;
    
    // Pending events per cell
    std::vector<event_queue<postsynaptic_spike_event> > cell_events_;
    
    // Time when the cell was last updated
    std::vector<time_type> last_time_updated_;
    
};
} // namespace mc
} // namespace nest
