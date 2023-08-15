NEURON {
    SUFFIX neuron_with_diffusion
	USEION s READ sd
}

PARAMETER {
	area : surface area of the CV (in µm^2, internal variable)
	diam : CV diameter (in µm, internal variable)
}

ASSIGNED {
	volume : volume of the CV (conversion factor between concentration and particle amount, in µm^3)
	sV : particle amount in the CV (in 1e-18 mol)
}

INITIAL {
	volume = area*diam/4 : = area*r/2 = 2*pi*r*h*r/2 = pi*r^2*h
	sV = sd * volume
}

BREAKPOINT {
	sV = sd * volume : read and normalize particle amount
}
