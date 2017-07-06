#pragma once
#include <vector>

// Model parameteres of leaky integrate and fire neuron model.
struct lif_cell_description {
    double tau_m = 10;    // membrane potential decaying constant [ms]
    double V_th = 10;     // firing threshold [mV]
    double C_m = 20;      // membrane capacity [pF]
    double E_L = 0;       // resting potential [mV]
    double V_m = E_L;     // membrane potential [mV]
    double V_reset = E_L; // reset potential [mV]
    double t_ref = 2;     // refractory period [ms]

    // incoming presynaptic synapses
    std::vector<double> synapse_weights;
};
