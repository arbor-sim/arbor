: Exponential current-based synapse

NEURON {
	POINT_PROCESS expsyn_curr
	RANGE w, tau, R_mem
	NONSPECIFIC_CURRENT I
}

UNITS {
	(ms) = (milliseconds)
	(mV) = (millivolt)
	(MOhm) = (megaohm)
}

PARAMETER {
	R_mem = 10.0 (MOhm) : membrane resistance
	tau = 5.0 (ms) : synaptic time constant
	w = 4.20075 (mV) : weight
}

STATE {
	g (mV) : instantaneous synaptic conductance
}

INITIAL {
	g = 0
}

BREAKPOINT {
	SOLVE state METHOD sparse : to match with expsyn_curr_calcium_plasticity

	I = -g / R_mem
}

DERIVATIVE state {
	: Exponential decay of postsynaptic potential
	g' = -g / tau
}

NET_RECEIVE(weight) {
	if (weight >= 0) {
		: Start of postsynaptic potential
		g = g + w
	}
}

