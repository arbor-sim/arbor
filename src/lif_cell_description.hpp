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
    
