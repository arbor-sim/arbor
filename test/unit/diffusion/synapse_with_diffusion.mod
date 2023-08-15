NEURON {
    POINT_PROCESS synapse_with_diffusion
	USEION s WRITE sd
}

PARAMETER {
	area : surface area of the CV (in µm^2, internal variable)
	diam : CV diameter (in µm, internal variable)
}

ASSIGNED {
	volume : volume of the CV (conversion factor between concentration and particle amount, in µm^3)
}

INITIAL {
	volume = area*diam/4 : = area*r/2 = 2*pi*r*h*r/2 = pi*r^2*h
}

BREAKPOINT {
}

NET_RECEIVE(weight) {
    sd = sd + weight * area / volume / 1000
}
