NEURON {
	POINT_PROCESS synapse_with_diffusion
	USEION s WRITE sd
}

PARAMETER {
	diam : CV diameter (in µm, internal variable)
    area : CV lateral surface (µm2)
}

STATE { alpha }

INITIAL { alpha = 0 }

BREAKPOINT {
	LOCAL volume
    volume = 0.25*area*diam
	sd = sd + alpha / volume
	alpha = 0
}

NET_RECEIVE(weight) { alpha = alpha + weight }
