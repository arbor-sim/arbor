#pragma once

#include <arbor/serdes.hpp>
#include <arbor/common_types.hpp>
#include <arbor/export.hpp>
#include <arbor/units.hpp>

namespace arb {

namespace U = arb::units;
using namespace U::literals;

// Model parameters of leaky integrate and fire neuron model.
struct ARB_SYMBOL_VISIBLE lif_cell {
    cell_tag_type source; // Label of source.
    cell_tag_type target; // Label of target.

    // Neuronal parameters.
    U::quantity tau_m = 10_ms;    // Membrane potential decaying constant [ms].
    U::quantity V_th  = 10_mV;    // Firing threshold [mV].
    U::quantity C_m   = 20_pF;    // Membrane capacitance [pF].
    U::quantity E_L   =  0_mV;    // Resting potential [mV].
    U::quantity E_R   =  0_mV;    // Reset potential [mV].
    U::quantity V_m   =  0_mV;    // Initial value of the Membrane potential [mV].
    U::quantity t_ref =  2_ms;    // Refractory period [ms].

    lif_cell() = default;
    lif_cell(cell_tag_type source, cell_tag_type target): source(std::move(source)), target(std::move(target)) {}

};

// LIF probe metadata, to be passed to sampler callbacks. Intentionally left blank.
struct ARB_SYMBOL_VISIBLE lif_probe_metadata {};

// Voltage estimate [mV].
// Sample value type: `double`
struct ARB_SYMBOL_VISIBLE lif_probe_voltage {};

} // namespace arb
