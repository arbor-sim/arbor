#!/usr/bin/env python
#coding: utf-8

import json
import math
import nrn_validation as V

try:
    from builtins import range
except ImportError:
    from __builtin__ import range

V.override_defaults_from_args()

# dendrite geometry: 100 µm long, varying diameter.
length = 100.0
npoints = 200
radius = lambda x: math.exp(-x)*(math.sin(40*x)*0.05+0.1)+0.1

xs = [float(i)/(npoints-1) for i in range(npoints)]
geom = [(length*x, 2.0*radius(x)) for x in xs]

model = V.VModel()
model.add_soma(12.6157)
model.add_dendrite('dend', geom)
model.add_iclamp(5, 80, 0.3, to='dend')

simdur = 100.0

data = V.run_nrn_sim(simdur, report_dt=10, model='ball_and_squiggle')
print(json.dumps(data))

V.nrn_stop()


