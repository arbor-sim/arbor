: Comment: LVA ca channel. Note: mtau is an approximation from the plots
: Reference:        Avery and Johnston 1996, tau from Randall 1997
: Comment: shifted by -10 mv to correct for junction potential
: Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON {
    SUFFIX Ca_LVA
    USEION ca READ eca WRITE ica
    RANGE gbar
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
    qt
}

STATE {
    m
    h
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ica = gbar*m*m*h*(v - eca)
}

DERIVATIVE states {
    LOCAL mInf, mRat, hInf, hRat
				  
    mInf = 1/(1 + exp((v + 40)/-6))
    hInf = 1/(1 + exp((v + 90)/6.4))
    mRat = qt/(5  + 20/(1 + exp((v + 35)/5)))				
    hRat = qt/(20 + 50/(1 + exp((v + 50)/7)))
     
    m' = (mInf - m)*mRat
    h' = (hInf - h)*hRat
}

INITIAL {
    qt = 2.3^((celsius-21)/10)
		   
    m = 1/(1 + exp((v + 40)/-6))
    h = 1/(1 + exp((v + 90)/6.4))
}
