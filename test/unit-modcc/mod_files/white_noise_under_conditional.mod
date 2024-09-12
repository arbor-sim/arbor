NEURON {
  POINT_PROCESS example
}

STATE {
  x
}

INITIAL {
  x = 10
}

BREAKPOINT {
  SOLVE state METHOD stochastic
}

WHITE_NOISE {
  xi
}

DERIVATIVE state {
  if (x > 2) {

  }
  else if (x > 1) {

  }
}
