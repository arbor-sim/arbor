TITLE Cardiac IKur  current and  nonspec cation current with identical kinetics

: Changes:
: derivimplicit -> cnexp
: Units set as CONSTANTS
: non-linear v

: Hodgkin - Huxley type channels, modified to fit IKur data from Feng et al Am J Physiol 1998 275:H1717 - H 1725
: Suffix from Kv15 to Kv1_5

NEURON {
    SUFFIX Kv1_5
    USEION k READ ek, ki, ko WRITE ik
    USEION na READ nai, nao
    USEION no WRITE ino VALENCE 1: nonspecific cation current
    RANGE gKur, ik, ino, Tauact, Tauinactf,Tauinacts, gnonspec, nao, nai, ko,ki
    RANGE minf, ninf, uinf, mtau , ntau, utau
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (mM) = (milli/liter)
:    F    = (faraday) (coulombs)
:    R    = (k-mole) (joule/degC)
}

PARAMETER {
    celsius (degC) : 37

    gKur      = 0.13195e-3 (S/cm2) :<0,1e9>
    Tauact    = 1          (ms)
    Tauinactf = 1          (ms)
    Tauinacts = 1          (ms)
    gnonspec  = 0          (S/cm2) :<0,1e9>

:    ek  :neuron
:    ki  :neuron
:    ko  :neuron
:    nai :neuron
:    nao :neuron
}

STATE {
     m n u
}

ASSIGNED {
    v       (mV)
    minf
    ninf
    uinf
    mtau    (ms)
    ntau    (ms)
    utau    (ms)

:    ino :neuron
:    ik  :neuron
}

CONSTANT {
    F = 96.485309
    R = 8.313424
}

INITIAL {
    rates(v, celsius)
    m = minf
    n = ninf
    u = uinf
}

BREAKPOINT {
:    LOCAL z, g :neuron

    SOLVE states METHOD cnexp

    LOCAL z, g :arbor

    z = (R*(celsius+273.15))/F
    g = (0.1 + 1/(1 + exp(-(v - 15)/13))) * m * m * m * n * u

    ik  = gKur     * g * (v - ek)
    ino = gnonspec * g * (v - z*log((nao+ko)/(nai+ki)))
}

DERIVATIVE states {    : exact when v held constant
    rates(v, celsius)
    m' = (minf - m)/mtau
    n' = (ninf - n)/ntau
    u' = (uinf - u)/utau
}

UNITSOFF
FUNCTION alp(v, i, celsius) {
    LOCAL q10 : order m n
:    v = v

    q10 = 2.2^((celsius - 37)/10)

    if (i==0) {
        alp = q10*0.65/(exp(-(v + 10)/8.5) + exp(-(v - 30)/59))
    } else if (i==1) {
        alp = 0.001*q10/(2.4 +10.9* exp(-(v + 90)/78))
    } else {
        alp = 0
    }
}

FUNCTION bet(v, i, celsius) {
    LOCAL q10 : order m n u
:    v = v

    q10 = 2.2^((celsius - 37)/10)

    if (i==0) {
        bet = q10*0.65/(2.5 + exp((v + 82)/17))
    } else if (i==1) {
        bet = q10*0.001*exp((v - 168)/16)
    } else {
        bet = 0
    }
}
                
FUNCTION ce(v,i) {   :  order m n u
:    v = v

    if (i==0) {
        ce = 1/(1 + exp(-(v + 30.3)/9.6))
    } else if (i==1) {
        ce = 1*(0.25+1/(1.35 + exp((v + 7)/14)))
    } else if (i==2) {
        ce = 1*(0.1+1/(1.1 + exp((v + 7)/14)))
    } else {
        ce = 0
    }
}


PROCEDURE rates(v, celsius) {
    LOCAL a, b, c :

    a = alp(v,0,celsius)
    b = bet(v,0,celsius)
    c = ce (v,0)
    mtau = 1/(a + b)/3*Tauact
    minf = c

    a = alp(v,1,celsius)
    b = bet(v,1,celsius)
    c = ce(v,1)
    ntau = 1/(a + b)/3*Tauinactf
    ninf = c

    c = ce(v,2)
    uinf = c
    utau = 6800*Tauinacts
}
UNITSON
