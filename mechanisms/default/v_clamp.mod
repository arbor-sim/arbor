NEURON {
    VOLTAGE_PROCESS v_clamp
    GLOBAL v0
}

PARAMETER {
    v0 = -60 (mV)
}

BREAKPOINT {
    v = v0
}
