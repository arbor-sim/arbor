NEURON {
	POINT_PROCESS Exp2Syn
	RANGE tau1, tau2, e
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau1=.1 (ms) : <1e-9,1e9>
	tau2 = 10 (ms) : <1e-9,1e9>
	e=0	(mV)
}

ASSIGNED {
	factor
}

STATE {
	A : (uS)
	B : (uS)
}

INITIAL {
	LOCAL tp
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = (B - A)*(v - e)
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

NET_RECEIVE(weight) {
	A = A + weight*factor
	B = B + weight*factor
}
