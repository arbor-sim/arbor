UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}
NEURON {
    SUFFIX pas
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

INITIAL {}

PARAMETER {
     g = .001 (S/cm2)
     e = -65  (mV) : we use -65 for the ball and stick model, instead of Neuron default of -70
}

STATE {
    s
}

ASSIGNED {
    v (mV)
}

BREAKPOINT {
    foo(i)
}

PROCEDURE foo(i) {
    if (g > 2) {
        if (g > 3) {
            i = 0
        } else {
            i = 1
        }
    } else {
        if (g < 1) {
            s = 2
        } else {
            rates(i)
        }
    }
}

PROCEDURE rates(i) {
LOCAL u
    if(i > 2) {
         u = 7
    } else {
         u = 5
         s = 42
    }
    s = u
}