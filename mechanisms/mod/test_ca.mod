: Example of mechanism that updates the concentration of an ionic species.

NEURON {
    SUFFIX test_ca
    USEION ca READ ica WRITE cai
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (molar) = (1/liter)
    (mM) = (millimolar)
    (um) = (micron)
}

PARAMETER {
    decay = 80 (ms)  : decay rate of calcium
    cai0 = 1e-4 (mM)
    factor =  2.59e-2 : gamma/2*F*depth
}

ASSIGNED {}

STATE {
    cai (mM)
}

BREAKPOINT  {
    SOLVE states METHOD cnexp
}

DERIVATIVE states   {
    cai' = (cai0 - cai)/decay - factor*ica
}
