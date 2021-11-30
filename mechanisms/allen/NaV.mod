TITLE Mouse sodium current
: Kinetics of Carter et al. (2012)
: Based on 37 degC recordings from mouse hippocampal CA1 pyramids

NEURON {
  SUFFIX NaV
  USEION na READ ena WRITE ina
  RANGE gbar
}

ASSIGNED {
    qt
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar	= .015  (S/cm2)
    celsius             (degC)

   : kinetic parameters
    Con		= 0.01  (/ms)                    : closed -> inactivated transitions
    Coff	= 40    (/ms)                    : inactivated -> closed transitions
    Oon		= 8     (/ms)                    : open -> Ineg transition
    Ooff	= 0.05  (/ms)                    : Ineg -> open transition
    alpha	= 400   (/ms)
    beta	= 12    (/ms)
    gamma	= 250   (/ms)                    : opening
    delta	= 60    (/ms)                    : closing

    alfac	= 2.51
    btfac	= 5.32

    : Vdep
    x1		= 24    (mV)                     : Vdep of activation (alpha)
    x2		= -24   (mV)                     : Vdep of deactivation (beta)
}

STATE {
    C1 FROM 0 TO 1
    C2 FROM 0 TO 1
    C3 FROM 0 TO 1
    C4 FROM 0 TO 1
    C5 FROM 0 TO 1
    I1 FROM 0 TO 1
    I2 FROM 0 TO 1
    I3 FROM 0 TO 1
    I4 FROM 0 TO 1
    I5 FROM 0 TO 1
    O  FROM 0 TO 1
    I6 FROM 0 TO 1
}

BREAKPOINT {
  SOLVE activation METHOD sparse
  ina = gbar*O*(v - ena)
}

INITIAL {
  qt = 2.3^((celsius-37)/10)
  SOLVE seqinitial
}

KINETIC activation {
  LOCAL f01, f02, f03, f04, f0O, f11, f12, f13, f14, fi1, fi2, fi3, fi4, fi5, fin, b01, b02, b03, b04, b0O, b11, b12, b13, b14, bi1, bi2, bi3, bi4, bi5, bin, ibtf

  ibtf = 1/btfac 

  f04 = qt*alpha*exp(v/x1)
  f03 = 2*f04
  f02 = 3*f04
  f01 = 4*f04
  f0O = qt*gamma

  f14 = alfac*f04
  f13 = 2*f14
  f12 = 3*f14
  f11 = 4*f14

  fi1 = qt*Con
  fi2 = fi1*alfac
  fi3 = fi2*alfac
  fi4 = fi3*alfac
  fi5 = fi4*alfac
  fin = qt*Oon

  b01 = qt*beta*exp(v/x2)
  b02 = 2*b01
  b03 = 3*b01
  b04 = 4*b01
  b0O = qt*delta

  b11 = b01*ibtf
  b12 = 2*b11
  b13 = 3*b11
  b14 = 4*b11

  bi1 = qt*Coff
  bi2 = bi1*ibtf
  bi3 = bi2*ibtf
  bi4 = bi3*ibtf
  bi5 = bi4*ibtf
  bin = qt*Ooff

  ~ C1 <-> C2    (f01, b01)
  ~ C2 <-> C3    (f02, b02)
  ~ C3 <-> C4    (f03, b03)
  ~ C4 <-> C5    (f04, b04)
  ~ C5 <-> O     (f0O, b0O)
  ~ O  <-> I6    (fin, bin)
  ~ I1 <-> I2    (f11, b11)
  ~ I2 <-> I3    (f12, b12)
  ~ I3 <-> I4    (f13, b13)
  ~ I4 <-> I5    (f14, b14)
  ~ I5 <-> I6    (f0O, b0O)
  ~ C1 <-> I1    (fi1, bi1)
  ~ C2 <-> I2    (fi2, bi2)
  ~ C3 <-> I3    (fi3, bi3)
  ~ C4 <-> I4    (fi4, bi4)
  ~ C5 <-> I5    (fi5, bi5)

  CONSERVE C1 + C2 + C3 + C4 + C5 + O + I1 + I2 + I3 + I4 + I5 + I6 = 1
}

: sets initial equilibrium
LINEAR seqinitial {
   LOCAL f01, f02, f03, f04, f11, f12, f13, f14, f0O, fi1, fi2, fi3, fi4, fi5, fin, b01, b02, b03, b04, b11, b12, b13, b14, b0O, bi1, bi2, bi3, bi4, bi5, bin, ibtf

  ibtf = 1/btfac
  f04 = qt*alpha*exp(v/x1)
  f03 = 2*f04
  f02 = 3*f04
  f01 = 4*f04
  f0O = qt*gamma

  f14 = alfac*f04
  f13 = 2*f14
  f12 = 3*f14
  f11 = 4*f14

  fi1 = qt*Con
  fi2 = fi1*alfac
  fi3 = fi2*alfac
  fi4 = fi3*alfac
  fi5 = fi4*alfac
  fin = qt*Oon

  b01 = qt*beta*exp(v/x2)
  b02 = 2*b01
  b03 = 3*b01
  b04 = 4*b01
  b0O = qt*delta

  b11 = b01*ibtf
  b12 = 2*b11
  b13 = 3*b11
  b14 = 4*b11

  bi1 = qt*Coff
  bi2 = bi1*ibtf
  bi3 = bi2*ibtf
  bi4 = bi3*ibtf
  bi5 = bi4*ibtf
  bin = qt*Ooff

  ~          I1*bi1 + C2*b01 - C1*(      fi1 + f01) = 0
  ~ C1*f01 + I2*bi2 + C3*b02 - C2*(b01 + fi2 + f02) = 0
  ~ C2*f02 + I3*bi3 + C4*b03 - C3*(b02 + fi3 + f03) = 0
  ~ C3*f03 + I4*bi4 + C5*b04 - C4*(b03 + fi4 + f04) = 0
  ~ C4*f04 + I5*bi5 +  O*b0O - C5*(b04 + fi5 + f0O) = 0
  ~ C5*f0O + I6*bin          -  O*(b0O + fin)       = 0
  ~          C1*fi1 + I2*b11 - I1*(      bi1 + f11) = 0
  ~ I1*f11 + C2*fi2 + I3*b12 - I2*(b11 + bi2 + f12) = 0
  ~ I2*f12 + C3*fi3 + I4*bi3 - I3*(b12 + bi3 + f13) = 0
  ~ I3*f13 + C4*fi4 + I5*b14 - I4*(b13 + bi4 + f14) = 0
  ~ I4*f14 + C5*fi5 + I6*b0O - I5*(b14 + bi5 + f0O) = 0

  ~ C1 + C2 + C3 + C4 + C5 + O + I1 + I2 + I3 + I4 + I5 + I6 = 1
}
