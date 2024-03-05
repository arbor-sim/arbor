NEURON { SUFFIX test_kinetic_under_conditional }

STATE { A B }

BREAKPOINT { : NOTE we need a newline here :/
  SOLVE foobar METHOD sparse
}

KINETIC foobar {
  LOCAL x, y

  x = 23*v
  y = 42*v

  if (v<0) {
    ~ A <-> B (x, y)
  }
  else {
    ~ A <-> B (y, x)
  }
}
