NEURON { SUFFIX test_derivative_alternating }

STATE { A B }

BREAKPOINT { : NOTE we need a newline here :/
  SOLVE foobar METHOD sparse
}

: NOTE this is OK to do, while alternating normal and reaction statements is not!

DERIVATIVE foobar {
  LOCAL x

  x = 23*v

  A' = B*x

  x = sin(x)

  B' = x*A
}
