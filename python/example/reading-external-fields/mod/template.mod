NEURON {
    POINT_PROCESS template
    RANGE xp, yp, zp, xd, yd, zd
    NONSPECIFIC_CURRENT i
}
    
PARAMETER {
    field
    xp yp zp : proximal coordinates
    xd yd zd : distal coordinates
}

ASSIGNED { da }

STATE {}

INITIAL {
    LOCAL dx, dy, dz
    : vector spanned by segment endopints
    dx = xd - xp
    dy = yd - yp
    dz = zd - zp
    : unit length of the segment
    da = 1.0/sqrt(dx*dx + dy*dy + dz*dz)
}

BREAKPOINT {
    LOCAL e
    : electrical field along x
    e = 42
    : induced current
    i = e*(xd - xp)*da
}


NET_RECEIVE(weight) {}
