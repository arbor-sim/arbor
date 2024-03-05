NEURON { SUFFIX test_misplaced_reaction }

STATE { A B }

BREAKPOINT { : NOTE we need a newline here :/
  SOLVE foobar METHOD sparse
}

: NOTE this is OK to do, while alternating normal and reaction statements is not!

DERIVATIVE foobar {
  LOCAL x

  ~ A <-> B (x, 1/x)
}
