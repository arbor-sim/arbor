#!/usr/bin/env python3
#coding: utf-8

import json
import nrn_validation as V

V.override_defaults_from_args()

model = V.VModel()

model.add_soma(18.8, Ra=100)
model.add_iclamp(10, 100, 0.1)

data = V.run_nrn_sim(100, report_dt=None, model='soma')
print(json.dumps(data))
V.nrn_stop()

