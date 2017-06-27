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
        
        last_time_updated_.resize(cells.size());
        
        spikes_.clear();
    }

    cell_kind get_cell_kind() const override {
        return cell_kind::lif_neuron;
    }
    
    // remove this after testing
    // samples voltage from time tstart to tend
    // with resolution given by class parameter
    // sampling_dt_
    void sample_voltage(value_type v, value_type E_L, value_type tau_m, time_type tstart, time_type tend, bool refractory_period) {
        
        time_type start = last_time_voltage_updated_;
        
        for(time_type time = last_time_voltage_updated_; time < tend; time += sampling_dt_) {
            if(!refractory_period)
                v *= exp(-(time - last_time_voltage_updated_)/tau_m);
            else {
                // if in refractory period, then at the moment of spike
                // let v has original value (which is >= V_thr)
                // otherwise, let v be equal to the resting potential E_L
                if (time > start)
                    v = E_L;
            }
            // put a pair of time and voltage
            voltage_.push_back({time, v});
            // update last time the voltage is updated
            last_time_voltage_updated_ = time;
        }
    }
    
    // advances a single cell (lid) with exact solution (jumps can be arbitrary)
    // parameter dt therefore ignored
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
            
            // remove this after testing
            // used just for sampling voltage
            if(sampling_dt_ > 0)
                sample_voltage(cell.V_m, cell.E_L, cell.tau_m, tstart, spike_time, false);
            
            // let the membrane potential decay
            cell.V_m *= exp(-(spike_time - tstart) / cell.tau_m);
            // add jump due to spike
            cell.V_m += weight/cell.C_m;
            
            tstart = spike_time;
            
            // if spike occured
            if(cell.V_m >= cell.V_th) {
                cell_member_type spike_neuron_gid = {gid_base_ + lid, 0};
                spike s = {spike_neuron_gid, spike_time};
                
                spikes_.push_back(s);
                
                // remove this after testing
                // used just for sampling voltage
                if(sampling_dt_ > 0)
                    sample_voltage(cell.V_m, cell.E_L, cell.tau_m, tstart, tstart + cell.t_ref, true);

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
        // advance each cell independently
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
        
        // REMOVE AFTER TESTING
        voltage_.clear();
        last_time_voltage_updated_ = 0;
    }
    
    
    // REMOVE AFTER TEST
    std::vector<std::pair<time_type, value_type> > voltage() {
        return voltage_;
    }
    
    // remove after testing
    // used to set up the resolution of sampling voltage
    void turn_on_sampling(time_type resolution) {
        sampling_dt_ = resolution;
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
    
    /* 
        REMOVE AFTER TEST
     */
    // sampling resolution
    time_type sampling_dt_ = 0;
    
    // assuming we have only 1 neuron in the group that
    // we want to track for the voltage
    std::vector<std::pair<time_type, value_type> > voltage_;
    
    // last time voltage was updated
    time_type last_time_voltage_updated_ = 0;
    
};
} // namespace mc
} // namespace nest
