NEURON { SUFFIX test_kinetic_alternating_differential }

STATE { A B C D }

BREAKPOINT { : NOTE we need a newline here :/
  SOLVE foobar METHOD sparse
}

KINETIC foobar {
  LOCAL x, y

  x = 23*v
  y = 42*v

  ~ A <-> B (x, y)
  C' = 0.1
  ~ C <-> D (x, y)
}
