NEURON {
  SUFFIX test4_kin_compartment
}

STATE {
	A
	B
	C
	d
	e
}

PARAMETER {
    x = 4
    y = 0.4
    z = 1.8
    w = 21
    s0 = 27
    s1 = 5.9
}

BREAKPOINT {
	SOLVE state METHOD sparse
}

INITIAL {
    A = 4.5
    B = 6.6
    C = 0.28
    d = 2
    e = 0

}

KINETIC state {
    COMPARTMENT s0 {A B C}
    COMPARTMENT s1 {d e}

    ~ A + B <-> C   ( x, y )
	~ A + d <-> e   ( z, w )
}