    NEURON {
        POINT_PROCESS efield
        RANGE xp, yp, zp, xd, yd, zd
        NONSPECIFIC_CURRENT i
    }
        
    PARAMETER {
        xp yp zp : proximal coordinates
        xd yd zd : distal coordinates
        e0       : amplitude
        omega    : frequency
    }
    
    ASSIGNED { da }
    
    STATE { t }
    
    INITIAL {
        LOCAL dx, dy, dz
        : vector spanned by segment endopints
        dx = xd - xp
        dy = yd - yp
        dz = zd - zp
        : unit length of the segment
        da = 1.0/sqrt(dx*dx + dy*dy + dz*dz)
        : time
        t = 0
    }
    
    BREAKPOINT {
        SOLVE state METHOD cnexp
        LOCAL e
        : electrical field along x
        e = e0 * sin(omega*t)
        : induced current
        i = e*(xd - xp)*da
    }
    
    DERIVATIVE state { t' = 1 }
    
    NET_RECEIVE(weight) {}
