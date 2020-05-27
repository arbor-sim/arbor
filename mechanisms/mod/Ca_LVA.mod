: Comment: LVA ca channel. Note: mtau is an approximation from the plots
: Reference:        Avery and Johnston 1996, tau from Randall 1997
: Comment: shifted by -10 mv to correct for junction potential
: Comment: corrected rates using q10 = 2.3, target temperature 34, orginal 21

NEURON  {
    SUFFIX Ca_LVA
    USEION ca READ eca WRITE ica
    RANGE gbar, g, ica
}

UNITS   {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER   {
    gbar = 0.00001 (S/cm2)
}

ASSIGNED    {
    v   (mV)
    g   (S/cm2)
    celsius (degC)
    mInf
    mTau
    hInf
    hTau
}

STATE   {
    m
    h
}

BREAKPOINT  {
    SOLVE states METHOD cnexp
    g = gbar*m*m*h
    ica = g*(v-eca)
}

DERIVATIVE states   {
    rates(v)
    m' = (mInf-m)/mTau
    h' = (hInf-h)/hTau
}

INITIAL{
    rates(v)
    m = mInf
    h = hInf
}

PROCEDURE rates(v){
  LOCAL qt
  qt = 2.3^((celsius-21)/10)

  mInf = 1/(1+ exp((v + 20)/-6))
  mTau = (5 + 20/(1+exp((v + 5)/5)))/qt
  hInf = 1/(1+ exp((v + 70)/6.4))
  hTau = (20 + 50/(1+exp((v + 30)/7)))/qt
}
