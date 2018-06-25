#pragma once

namespace arb {

// Model parameteres of leaky integrate and fire neuron model.
struct lif_cell {
    // Neuronal parameters.
    double tau_m = 10;    // Membrane potential decaying constant [ms].
    double V_th = 10;     // Firing threshold [mV].
    double C_m = 20;      // Membrane capacitance [pF].
    double E_L = 0;       // Resting potential [mV].
    double V_m = E_L;     // Initial value of the Membrane potential [mV].
    double V_reset = E_L; // Reset potential [mV].
    double t_ref = 2;     // Refractory period [ms].
};

} // namespace arb
