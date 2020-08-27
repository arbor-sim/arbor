#!/usr/bin/env python3
#coding: utf-8

import json
import nrn_validation as V

V.override_defaults_from_args()

# dendrite geometry: 200 µm long, 1 µm diameter.
geom = [(0,1), (200, 1)]

model = V.VModel()
model.add_soma(12.6157)
model.add_dendrite('dend', geom)
model.add_exp2_syn('dend')

model.add_spike(10, 0.04)
model.add_spike(20, 0.04)
model.add_spike(40, 0.04)

data = V.run_nrn_sim(70, report_dt=10, model='exp2syn')
print(json.dumps(data))
V.nrn_stop()

