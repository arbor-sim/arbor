NEURON {
	POINT_PROCESS synapse_with_diffusion
	USEION s WRITE sd
}

PARAMETER {
	diam : CV diameter (in µm, internal variable)
}

NET_RECEIVE(weight) {
	sd = sd + 4 * 0.001 * weight / diam
}
