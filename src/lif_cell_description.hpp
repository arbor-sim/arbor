#pragma once

// Model parameteres of leaky integrate and fire neuron model.
struct lif_cell_description {
    // Neuronal parameters.
    double tau_m = 10;    // Membrane potential decaying constant [ms].
    double V_th = 10;     // Firing threshold [mV].
    double C_m = 20;      // Membrane capacity [pF].
    double E_L = 0;       // Resting potential [mV].
    double V_m = E_L;     // Membrane potential [mV].
    double V_reset = E_L; // Reset potential [mV].
    double t_ref = 2;     // Refractory period [ms].

    // External Poisson input parameters.
    int n_poiss = 0;      // Number of external Poisson neurons.
    double rate = 1;      // Rate of each Poisson neuron [kHz].
    float w_poiss = 1;    // Weight of each Poisson->LIF synapse.
    float d_poiss = 1;    // Delay of each Poisson->LIF synapse.
};
