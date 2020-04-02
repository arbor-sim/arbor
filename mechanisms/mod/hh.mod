NEURON {
    SUFFIX hh
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    NONSPECIFIC_CURRENT il
    RANGE gnabar, gkbar, gl, el, gna, gk
    GLOBAL minf, hinf, ninf, mtau, htau, ntau
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gnabar = .12 (S/cm2)
    gkbar = .036 (S/cm2)
    gl = .0003 (S/cm2)
    el = -54.3 (mV)
    celsius
}

STATE {
    m h n
}

ASSIGNED {
    v (mV)

    gna (S/cm2)
    gk (S/cm2)
    minf
    hinf
    ninf
    mtau (ms)
    htau (ms)
    ntau (ms)
    q10
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gna = gnabar*m*m*m*h
    ina = gna*(v - ena)
    gk = gkbar*n*n*n*n
    ik = gk*(v - ek)
    il = gl*(v - el)
}

INITIAL {
    q10 = 3^((celsius - 6.3)/10)
    rates(v)
    m = minf
    h = hinf
    n = ninf
}

DERIVATIVE states {
    rates(v)
    m' = (minf-m)/mtau
    h' = (hinf-h)/htau
    n' = (ninf-n)/ntau
}

PROCEDURE rates(v)
{
    LOCAL  alpha, beta, sum

    :"m" sodium activation system
    alpha = .1 * vtrap(-(v+40),10)
    beta =  4 * exp(-(v+65)/18)
    sum = alpha + beta
    mtau = 1/(q10*sum)
    minf = alpha/sum

    :"h" sodium inactivation system
    alpha = .07 * exp(-(v+65)/20)
    beta = 1 / (exp(-(v+35)/10) + 1)
    sum = alpha + beta
    htau = 1/(q10*sum)
    hinf = alpha/sum

    :"n" potassium activation system
    alpha = .01*vtrap(-(v+55),10)
    beta = .125*exp(-(v+65)/80)
    sum = alpha + beta
    ntau = 1/(q10*sum)
    ninf = alpha/sum
}

FUNCTION vtrap(x,y) {
    : use built in exprelr(z) = z/(exp(z)-1), which handles the z=0 case correctly
    vtrap = y*exprelr(x/y)
}

