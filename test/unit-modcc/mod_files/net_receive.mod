NEURON {
  POINT_PROCESS net_rec_tst
  NONSPECIFIC_CURRENT i
}

PARAMETER { foo e }

ASSIGNED { g }

INITIAL { g = 0 }

BREAKPOINT {
  i = g*(v - e)
}

NET_RECEIVE(w, foo) {
     g = g + foo
}
