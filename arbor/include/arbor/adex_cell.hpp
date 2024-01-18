#pragma once

#include <arbor/serdes.hpp>
#include <arbor/common_types.hpp>
#include <arbor/export.hpp>

namespace arb {

// Model parameters of adaptive exponential cell (AdEx) see
//
// Self-sustained asynchronous irregular states and Up–Down states in thalamic,
// cortical and thalamocortical networks of nonlinear integrate-and-fire neurons
//
// A. Destehxe 2009
//
//
struct ARB_SYMBOL_VISIBLE adex_cell {
    cell_tag_type source; // Label of source
    cell_tag_type target; // Label of target

    // Neuronal parameters.
    double delta =   2.5;   // Steepness parameter [mV]
    double V_th  = -20;     // Firing threshold [mV]
    double C_m   =   0.28;  // Membrane capacitance [pF]
    double E_L   = -70;     // Resting potential [mV]
    double E_R   = E_L;     // Reset potential [mV]
    double V_m   = E_L;     // Initial value of the Membrane potential [mV]
    double t_ref =   2.5;   // Refractory period [ms]
    double g     =   0.03;     // Leak conductivity [uS]
    // Adaption parameters
    double tau   = 144;     // Adaption decaying constant [ms]
    double w     =   0;     // Initial value for adaption parameter [nA]
    double a     =   0.004; // Adaption dynamics [uS].
    double b     =   0.08;  // When spikes trigger, increase w by this [nA]
    adex_cell() = default;
    adex_cell(cell_tag_type source, cell_tag_type target): source(std::move(source)), target(std::move(target)) {}

    ARB_SERDES_ENABLE(adex_cell, source, target, delta, V_th, C_m, E_L, E_R, V_m, t_ref);
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
