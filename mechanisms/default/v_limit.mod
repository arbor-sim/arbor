NEURON {
    VOLTAGE_PROCESS v_limit
    GLOBAL v_low, v_high
}

PARAMETER {
    v_high =  20 (mV)
    v_low  = -70 (mV)
}

BREAKPOINT {
     v = max(min(v, v_high), v_low)
}
