#!/usr/bin/env python3
#coding: utf-8

import json
import nrn_validation as V

V.override_defaults_from_args()

# dendrite geometry: all 100 µm long, 1 µm diameter.
geom = [(0,1), (200, 1)]

model = V.VModel()
model.add_soma(12.6157)
model.add_dendrite('dend', geom)
model.add_iclamp(5, 80, 0.3, to='dend')

data = V.run_nrn_sim(100, report_dt=10, model='ball_and_stick')
print(json.dumps(data))
V.nrn_stop()

