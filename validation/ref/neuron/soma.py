#!/usr/bin/env python
#coding: utf-8

import json
import nrn_validation as V

V.override_defaults_from_args()

# dendrite geometry: all 100 µm long, 1 µm diameter.
geom = [(0,1), (100, 1)]

model = V.VModel()

model.add_soma(18.8, Ra=100)
model.add_iclamp(10, 100, 0.1)

# NB: this doesn't seem to have converged with
# the default dt of 0.001.
data = V.run_nrn_sim(100, report_dt=None, model='soma')
print json.dumps(data)
V.nrn_stop()

