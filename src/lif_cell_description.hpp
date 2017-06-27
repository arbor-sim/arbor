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
    // MODEL PARAMETERS OF SIMPLE
    // LEAKY INTEGRATE AND FIRE NEURON MODEL
    ////////////////////////////////////////////////
    
    using index_type = nest::mc::cell_lid_type;
    using value_type = double;
    
    value_type tau_m = 20;              // membrane potential decaying constant (ms)
    value_type V_th = 10;               // firing threshold (mV)
    value_type C_m = 20;                // membrane capacity (pF)
    value_type E_L = 0;                 // resting potential (mV)
    value_type V_m = E_L;               // membrane potential (mV)
    value_type V_reset = E_L;           // reset potential (mV)
    value_type t_ref = 2;               // refractory period (ms)
  
    // incoming presynaptic synapses
    std::vector<value_type> synapse_weights;

};
    
