#pragma once

#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <common_types.hpp>
#include <cell_tree.hpp>
#include <util/debug.hpp>
#include <util/pprintf.hpp>
#include <util/rangeutil.hpp>


struct lif_cell_description {
    ////////////////////////////////////////////////
    // MODEL PARAMETERS TAKEN FROM NEST
    // http://www.nest-simulator.org/cc/iaf_neuron/
    ////////////////////////////////////////////////
    
    using index_type = nest::mc::cell_lid_type;
    using value_type = double;
    
    value_type V_m = 0;                 // membrane potential (mV)
    value_type tau_m = 20;              // membrane potential decaying constant (ms)
    value_type V_th = -50;              // firing threshold (mV)
    value_type C_m = 20;                // membrane capacity (pF)
    value_type E_L = 0;                 // resting potential (mV)
    value_type V_reset = -65;           // reset potential (mV)
    value_type I_e = 0;                 // constant external input current (pA)
    value_type tau_s = 2;               // peak time for alpha-function (ms)
    value_type t_ref = 2;               // refractory period (ms)
    value_type I_syn = 0;               // input synaptic current (pA)
    
    std::vector<value_type> synapse_weights;

};
    
/*
    
using index_type = cell_lid_type;

/// high-level abstract representation of a lif cell
class lif_cell {
public:
    
    // constructor with default values
    lif_cell();
    
    ////////////////////////////////////////////////
    // SYNAPSES
    ////////////////////////////////////////////////
    struct synapse_instance {
        // TODO: set default weight
        double weight = 0.1;
        
        synapse_instance(){};
        
        synapse_intance(double w):weight{w};
        
        // TODO: implement the dynamics of synapse
        // for the time being, weights are static
        double update_synapse(double tfinal) {
            return weight;
        }
    };
    
    void add_synapse(double w)
    {
        synapses_.push_back(synapse_instance(w));
    }
    
    const std::vector<synapse_instance>& synapses() const {
        return synapses_;
    }
    
    ////////////////////////////////////////////////
    // MEMBRANE DYNAMICS
    ////////////////////////////////////////////////
    
    // evolves the state of neuron
    void advance(double tfinal, double dt);
    
    // enqueues incoming spike event
    void enqueue_event(double spike_time, double weight w);
    
    // returns firing times of current neuron
    std::vector<double> get_spikes();
    
    // clears spikes of current neuron
    void clear_spikes();
    
    ////////////////////////////////////////////////
    // GETTERS AND SETTERS
    ////////////////////////////////////////////////
    
    // TODO: make this more elegant
    // with parameter_name and value
    // instead of a bunch of setters/getters
    
    void set_voltage(double v_);
    void set_tau_m(double tau_m_);
    void set_V_thr(double V_thr_);
    void set_C_m(double C_m_);
    void set_E_L(double E_L_);
    void set_V_reset(double V_reset_);
    void set_tau_syn(double tau_syn_);
    void set_I_e(double I_e_);
    void set_tau_s(double tau_s_);
    
    
    double get_voltage();
    double get_tau_m();
    double get_V_thr();
    double get_C_m();
    double get_E_L();
    double get_V_reset();
    double get_tau_syn();
    double get_I_e();
    double get_tau_s();
    
    
private:
    // storage for connections
    std::vector<index_type> parents_;
    
    // the synapses
    std::vector<synapse_instance> synapses_;
    
    // contains firing times of this neuron
    // (since last invokation of clear_spikes())
    std::vector<double> spikes_;
    
    // contains incoming spikes that need to be processed
    std::vector<postsynaptic_spike_event>& events_;
    
    // last time neuron state was updated
    double time = 0;
    
    ////////////////////////////////////////////////
    // MODEL PARAMETERS TAKEN FROM NEST
    // http://www.nest-simulator.org/cc/iaf_neuron/
    ////////////////////////////////////////////////
    
    double V_m = 0;                 // membrane potential (mV)
    double tau_m = 20;              // membrane potential decaying constant (ms)
    double V_thr = 20;              // firing threshold (mV)
    double C_m = 20;                // membrane capacity (pF)
    double E_L = 0;                 // resting potential (mV)
    double V_reset = -65;           // reset potential (mV)
    double tau_syn = 5;             // alpha-function rise time (ms)
    double I_e = 0;                 // constant external input current (pA)
    double tau_s = 2;               // peak time for alpha-function (ms)

    
} // namespace mc
} // namespace nest
 */
