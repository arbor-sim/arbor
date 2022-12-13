NEURON {
    VOLTAGE_PROCESS vmech
}

INITIAL { c = 0 }

STATE { c }

PARAMETER {
    tau_c = 150 (ms)
}

BREAKPOINT { v = 42 }
