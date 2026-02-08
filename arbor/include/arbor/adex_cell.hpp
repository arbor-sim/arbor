#pragma once

#include <arbor/serdes.hpp>
#include <arbor/common_types.hpp>
#include <arbor/export.hpp>
#include <arbor/units.hpp>

namespace arb {

// Model parameters of adaptive exponential cell (AdEx) see
//
// Self-sustained asynchronous irregular states and Up–Down states in thalamic,
// cortical and thalamocortical networks of nonlinear integrate-and-fire neurons
//
// A. Destehxe 2009

namespace U = arb::units;
using namespace arb::units::literals;


struct ARB_SYMBOL_VISIBLE adex_cell {
    cell_tag_type source; // Label of source
    cell_tag_type target; // Label of target

    // Neuronal parameters.
    U::quantity delta =   2.5_mV;   // Steepness parameter [mV]
    U::quantity V_th  =  20_mV;     // Firing threshold [mV]
    U::quantity C_m   =   0.28_nF;  // Membrane capacitance [nF]
    U::quantity E_L   = -70_mV;     // Resting potential [mV]
    U::quantity E_R   = -70_mV;     // Reset potential [mV]
    U::quantity V_m   = -70_mV;     // Initial value of the Membrane potential [mV]
    U::quantity t_ref =   2.5_ms;   // Refractory period [ms]
    U::quantity g     =   0.03_uS;  // Leak conductivity [uS]
    // Adaption parameters
    U::quantity tau   = 144_ms;     // Adaption decaying constant [ms]
    U::quantity w     =   5.0_pA;   // Initial value for adaption parameter [nA]
    U::quantity a     =   0.004_uS; // Adaption dynamics [uS].
    U::quantity b     =   0.08_nA;  // When spikes trigger, increase w by this [nA]
};

// ADEX probe metadata, to be passed to sampler callbacks. Intentionally left blank.
struct ARB_SYMBOL_VISIBLE adex_probe_metadata {};

// Voltage estimate `U` [mV].
// Sample value type: `double`
struct ARB_SYMBOL_VISIBLE adex_probe_voltage {};

// Adapation variable `w` [nA].
// Sample value type: `double`
struct ARB_SYMBOL_VISIBLE adex_probe_adaption {};

} // namespace arb
