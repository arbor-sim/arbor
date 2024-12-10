: Based on Im model of Vervaeke et al. (2006)

NEURON   {
   SUFFIX Im_v2
   USEION k READ ek WRITE ik
   RANGE gbar, qt
}

UNITS {
   (S) =  (siemens)
   (mV) = (millivolt)
   (mA) = (milliamp)
}

PARAMETER {
   celsius          (degC)     
   gbar = 0.00001   (S/cm2)
}

ASSIGNED { qt }
STATE { m }

BREAKPOINT {
   SOLVE states METHOD cnexp
   ik = gbar*m*(v - ek)
}

DERIVATIVE states {
    LOCAL mAlpha, mBeta, mAlphaBeta

    mAlpha     = m_alpha(v)
    mBeta      = m_beta(v)
    mAlphaBeta = mAlpha + mBeta

    m' = qt*(mAlpha - m*mAlphaBeta)/(1 + 15*mAlphaBeta)
}

INITIAL {
    LOCAL mAlpha, mBeta
    
    mAlpha = m_alpha(v)
    mBeta  = m_beta(v)
    m = mAlpha/(mAlpha + mBeta)
    qt = 2.3^((celsius - 30)/10)
}

FUNCTION m_alpha(v) { m_alpha = 0.007*exp(( 6 *      0.4  * (v + 48))/26.12) }
FUNCTION m_beta(v)  { m_beta  = 0.007*exp((-6 * (1 - 0.4) * (v + 48))/26.12) }
