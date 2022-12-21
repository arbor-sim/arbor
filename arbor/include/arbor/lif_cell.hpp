#pragma once

#include <arbor/common_types.hpp>
#include <arbor/export.hpp>

namespace arb {

// Model parameters of leaky integrate and fire neuron model.
struct ARB_SYMBOL_VISIBLE lif_cell {
    cell_tag_type source; // Label of source.
    cell_tag_type target; // Label of target.

    // Neuronal parameters.
    double tau_m = 10;    // Membrane potential decaying constant [ms].
    double V_th = 10;     // Firing threshold [mV].
    double C_m = 20;      // Membrane capacitance [pF].
    double E_L = 0;       // Resting potential [mV].
    double E_R = E_L;     // Reset potential [mV].
    double V_m = E_L;     // Initial value of the Membrane potential [mV].
    double t_ref = 2;     // Refractory period [ms].

    lif_cell() = delete;
    lif_cell(cell_tag_type source, cell_tag_type target): source(std::move(source)), target(std::move(target)) {}
};

// LIF probe metadata, to be passed to sampler callbacks. Intentionally left blank.
struct ARB_SYMBOL_VISIBLE lif_probe_metadata {};

// Voltage estimate [mV].
// Sample value type: `double`
struct ARB_SYMBOL_VISIBLE lif_probe_voltage {};

} // namespace arb
