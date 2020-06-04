: Comment: LVA ca channel. Note: mtau is an approximation from the plots
: Reference:        Avery and Johnston 1996, tau from Randall 1997
: Comment: shifted by -10 mv to correct for junction potential
: Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON {
    SUFFIX Ca_LVA
    USEION ca READ eca WRITE ica
    RANGE gbar, g
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 0.00001 (S/cm2)
    celsius
}

ASSIGNED {
    g (S/cm2)
    mInf
    mTau
    hInf
    hTau
}

STATE {
    m
    h
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar*m*m*h
    ica = g*(v-eca)
}

DERIVATIVE states {
    rates(v, celsius)
    m' = (mInf-m)/mTau
    h' = (hInf-h)/hTau
}

INITIAL {
    rates(v, celsius)
    m = mInf
    h = hInf
}

PROCEDURE rates(v, celsius) {
  LOCAL qt
  qt = 2.3^((celsius-21)/10)

	UNITSOFF
		v = v + 10
		mInf = 1.0000/(1+ exp((v - -30.000)/-6))
		mTau = (5.0000 + 20.0000/(1+exp((v - -25.000)/5)))/qt
		hInf = 1.0000/(1+ exp((v - -80.000)/6.4))
		hTau = (20.0000 + 50.0000/(1+exp((v - -40.000)/7)))/qt
		v = v - 10
	UNITSON
}
