: Dynamics that track inside calcium concentration
: modified from Destexhe et al. 1994

NEURON {
    SUFFIX CaDynamics
    USEION ca READ ica WRITE cai
    RANGE decay, gamma, minCai, depth
    GLOBAL F
}

UNITS {
    (mV)    = (millivolt)
    (mA)    = (milliamp)
    (molar) = (1/liter)
    (mM)    = (millimolar)
    (um)    = (micron)
}

PARAMETER {
    F        = 96485.3321233100184  (coulomb/mole) : Faraday's constant
    gamma    = 0.05                 ()             : percent of free calcium (not buffered)
    decay    = 80                   (ms)           : rate of removal of calcium
    depth    = 0.1                  (um)           : depth of shell
    minCai   = 1e-4                 (mM)
}

INITIAL {
    cai = minCai
}

STATE { cai (mM) }

BREAKPOINT {
    SOLVE states METHOD cnexp
}

DERIVATIVE states {
    cai' = -10000*ica*gamma/(2*F*depth) - (cai - minCai)/decay
}
