NEURON { SUFFIX test_kinetic_alternating_reaction }

STATE { A B C D }

BREAKPOINT { : NOTE we need a newline here :/
  SOLVE foobar METHOD sparse
}

KINETIC foobar {
  LOCAL x, y

  x = 23*v
  y = 42*v

  ~ A <-> B (x, y)

  x = sin(y)
  y = cos(x)

  ~ C <-> D (x, y)
}
