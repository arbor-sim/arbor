TITLE K-A
: K-A current for Mitral Cells from Wang et al (1996)
: M.Migliore Jan. 2002

NEURON {
	SUFFIX kamt
	USEION k READ ek WRITE ik
	RANGE  gbar
	GLOBAL minf, mtau, hinf, htau
}

PARAMETER {
	gbar = 0.002   	(mho/cm2)	
								
	celsius
	ek		(mV)            : must be explicitly def. in hoc
	v 		(mV)
	a0m=0.04
	vhalfm=-45
	zetam=0.1
	gmm=0.75

	a0h=0.018
	vhalfh=-70
	zetah=0.2
	gmh=0.99

	sha=9.9
	shi=5.7
	
	q10=3
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ik 		(mA/cm2)
	minf 		mtau (ms)	 	
	hinf 		htau (ms)	 	
}
 

STATE { m h}

BREAKPOINT {
        SOLVE states METHOD cnexp
	ik = gbar*m*h*(v - ek)
} 

INITIAL {
	trates(v)
	m=minf  
	h=hinf  
}

DERIVATIVE states {   
        trates(v)      
        m' = (minf-m)/mtau
        h' = (hinf-h)/htau
}

PROCEDURE trates(v) {  
	LOCAL qt
        qt=q10^((celsius-24)/10)
        minf = 1/(1 + exp(-(v-sha-7.6)/14))
	mtau = betm(v)/(qt*a0m*(1+alpm(v)))

        hinf = 1/(1 + exp((v-shi+47.4)/6))
	htau = beth(v)/(qt*a0h*(1+alph(v)))
}

FUNCTION alpm(v(mV)) {
  alpm = exp(zetam*(v-vhalfm)) 
}

FUNCTION betm(v(mV)) {
  betm = exp(zetam*gmm*(v-vhalfm)) 
}

FUNCTION alph(v(mV)) {
  alph = exp(zetah*(v-vhalfh)) 
}

FUNCTION beth(v(mV)) {
  beth = exp(zetah*gmh*(v-vhalfh)) 
}
