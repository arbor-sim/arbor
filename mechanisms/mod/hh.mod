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
    celsius = 6.3 (degC)
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
    rates(v)
    m = minf
    h = hinf
    n = ninf
}

DERIVATIVE states {
    rates(v)
    m' =  (minf-m)/mtau
    h' = (hinf-h)/htau
    n' = (ninf-n)/ntau
}

PROCEDURE rates(v)
{
    LOCAL  alpha, beta, sum, q10

    q10 = 3^((celsius - 6.3)/10)

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

: We don't trap for zero in the denominator like Neuron, because function
: inlining in modparser won't support it. There is a good argument that
: vtrap should be provided as a built in of the language, because
:   - it is a common pattern in many mechanisms.
:   - it can be implemented efficiently on different back ends if the
:     compiler has enough information.
FUNCTION vtrap(x,y) {
    vtrap = x/(exp(x/y) - 1)
}

