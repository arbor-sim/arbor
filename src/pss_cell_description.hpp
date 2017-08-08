#pragma once

// Cell spiking according to Poisson distribution with given rate [kHz]
struct pss_cell_description {
    // lambda = 1 / rate_kHz
    // If rate_kHz = 10kHz then lambda_ms = 0.1ms.
    // Represents the expected inter-spike time period.
    double lambda = 0.2;

    pss_cell_description() = default;

    // Constructor taking rate_kHz [kHz] as input and computing lambda_ms = 1 / rate_kHz [ms]
    pss_cell_description(double rate_kHz) {
        // Make sure that rate_kHz > 0
        if (rate_kHz <= 0.0) {
            std::out_of_range("Spiking rate of Poisson neuron must be strictly positive!");
        } else {
            lambda = 1.0/rate_kHz;
        }
    }
};
