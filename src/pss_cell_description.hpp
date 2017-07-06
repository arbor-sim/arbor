#pragma once

// Cell spiking according to Poisson distribution with given rate [kHz]
struct pss_cell_description {
    // lambda = 1 / rate_kHz
    // If rate_kHz = 10kHz then lambda_ms = 0.1ms.
    // Represents the expected inter-spike time period.
    double lambda = 0.2;

    pss_cell_description() {}

    // Constructor taking rate_kHz [kHz] as input and computing lambda_ms = 1 / rate_kHz [ms]
    pss_cell_description(double rate_kHz) {
        // Check if rate_kHz is <= 0, but pay attention that its of type double
        if (rate_kHz < 1e-7) {
            std::logic_error("Spiking rate of Poisson neuron cannot be zero!");
        } else {
            lambda = 1.0/rate_kHz;
        }
    }
};
