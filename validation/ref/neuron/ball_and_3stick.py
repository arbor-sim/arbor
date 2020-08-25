#!/usr/bin/env python3
#coding: utf-8

import json
import nrn_validation as V

V.override_defaults_from_args()

# dendrite geometry: all 100 µm long, 1 µm diameter.
geom = [(0,1), (100, 1)]

model = V.VModel()
model.add_soma(12.6157)
model.add_dendrite('dend1', geom)
model.add_dendrite('dend2', geom, to='dend1')
model.add_dendrite('dend3', geom, to='dend1')

model.add_iclamp(5, 80, 0.45, to='dend2')
model.add_iclamp(40, 10, -0.2, to='dend3')

simdur = 100.0

data = V.run_nrn_sim(simdur, report_dt=10, model='ball_and_3stick')
print(json.dumps(data))

V.nrn_stop()

