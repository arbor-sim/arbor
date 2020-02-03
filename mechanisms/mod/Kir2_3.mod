TITLE Cerebellum Granule Cell Model

: Changes:
: derivimplicit -> cnexp
: celsius needs to be initialized in Arbor


:COMMENT
:    Reference: Theta-Frequency Bursting and Resonance in Cerebellar Granule Cells:Experimental
:    Evidence and Modeling of a Slow K+-Dependent Mechanism
:    Egidio D'Angelo,Thierry Nieus,Arianna Maffei,Simona Armano,Paola Rossi,Vanni Taglietti,
:    Andrea Fontana and Giovanni Naldi
:
:Suffix from Ubc_Kir to Kir2_3
:ENDCOMMENT
 
NEURON { 
    SUFFIX Kir2_3
    USEION k READ ek WRITE ik 
    RANGE gkbar, ik, g, alpha_d, beta_d, ek
    RANGE Aalpha_d, Kalpha_d, V0alpha_d
    RANGE Abeta_d, Kbeta_d, V0beta_d
    RANGE d_inf, tau_d 
} 
 
UNITS { 
    (mA) = (milliamp) 
    (mV) = (millivolt) 
} 
 
PARAMETER { 
    Aalpha_d  = 0.13289  (/ms)
    Kalpha_d  = -24.3902 (mV)

    V0alpha_d = -83.94   (mV)
    V0beta_d  = -83.94   (mV)

    Abeta_d   = 0.16994  (/ms)
    Kbeta_d   = 35.714   (mV)

    gkbar = 0.0009       (mho/cm2)

    v  (mV)
    celsius = 30 (degC)

:    ek :neuron
} 

STATE { 
    d 
} 

ASSIGNED { 
    d_inf
    tau_d   (ms)
    g       (mho/cm2)
    alpha_d (/ms) 
    beta_d  (/ms)

:    ik :neuron
} 
 
INITIAL { 
    rate(v, celsius)
    d = d_inf 
} 
 
BREAKPOINT { 
    SOLVE states METHOD cnexp
    g = gkbar*d   
    ik = g*(v - ek) 
    alpha_d = alp_d(v, celsius)
    beta_d  = bet_d(v, celsius)
} 
 
DERIVATIVE states { 
    rate(v, celsius)
    d' =(d_inf - d)/tau_d 
} 
 
FUNCTION alp_d(v, celsius) {
    LOCAL Q10
    Q10   = 3^( (celsius-20)/10 )
    alp_d = Q10*Aalpha_d*exp((v-V0alpha_d)/Kalpha_d) 
} 
 
FUNCTION bet_d(v, celsius) {
    LOCAL Q10
    Q10 = 3^( (celsius-20 )/10 )
    bet_d = Q10*Abeta_d*exp((v-V0beta_d)/Kbeta_d) 
} 
 
PROCEDURE rate(v, celsius) {
    LOCAL a_d, b_d
    a_d = alp_d(v, celsius)
    b_d = bet_d(v, celsius)
    tau_d = 1/(a_d + b_d) 
    d_inf = a_d/(a_d + b_d) 
}