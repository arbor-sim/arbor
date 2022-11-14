: Leaky-integrate and fire neuron

: If the threshold potential is crossed, the membrane resistance is set to a very low value
: to pull the membrane potential towards the reset potential for the duration of the refractory period.
: After that, the membrane resistance is set to the comparably higher leak resistance again.
: The threshold potential of Arbor's threshold detector should match the threshold potential here.

: authors: Sebastian Schmitt, Jannik Luboeinski
: based on https://github.com/tetzlab/FIPPA/blob/main/STDP/mechanisms/lif.mod

NEURON {
	SUFFIX lif
	NONSPECIFIC_CURRENT i
	RANGE R_reset, R_leak, V_reset, V_rev, V_th, I_0, i_factor, t_ref
}

UNITS {
	(ms) = (milliseconds)
	(mV) = (millivolt)
	(MOhm) = (megaohm)
}

STATE {
	refractory_counter
}

INITIAL {
	refractory_counter = t_ref + 1 : start not refractory
		
	v = V_rev : membrane potential
}

PARAMETER {
	R_leak = 10 (MOhm) : membrane resistance during leakage (standard case)
	R_reset = 1e-10 (MOhm) : membrane resistance during refractory period (very small to yield fast reset)
	I_0 = 0 : constant input current (in nA)
	i_factor = 1 : conversion factor from nA to mA/cm^2; for point neurons

	V_reset = -70 (mV) : reset potential
	V_rev = -65 (mV) : reversal potential
	V_th = -55 (mV) : threshold potential to be crossed for spiking
	t_ref = 2 (ms) : refractory period
}

BREAKPOINT {
	SOLVE state METHOD cnexp

	LOCAL R_mem
	LOCAL E
	
	: threshold crossed -> start refractory counter
	if (v > V_th) {
	   refractory_counter = 0
	}

	: choose between leak and reset potential
	if (refractory_counter <= t_ref) { : in refractory period - strong drive of v towards V_reset
		R_mem = R_reset
		E = V_reset
	} else { : outside refractory period
		R_mem = R_leak
		E = V_rev
	}

	i = (((v - E) / R_mem) - I_0) * i_factor : current density in units of mA/cm^2
}

DERIVATIVE state {
	if (refractory_counter <= t_ref) {
		refractory_counter' = 1
	}	
}
