#pragma once

// Cell spiking according to Poisson distribution with given rate
struct pss_cell_description {
    // lambda = 1 / rate
    // If rate = 10Hz then lambda = 0.1s.
    // Represents the expected inter-spike time period.
    double lambda = 0.2;
    
    pss_cell_description() {}
    
    // Constructor taking rate [Hz] as input and computing lambda=1/rate [s]
    pss_cell_description(double rate) {
        if (rate == 0) {
            std::logic_error("Spiking rate of Poisson neuron cannot be zero!");
        } else {
            lambda = 1.0/rate;
        }
    }
};
