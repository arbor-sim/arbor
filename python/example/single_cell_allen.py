#!/usr/bin/env python3

import arbor as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from dataclasses import dataclass

# Check arbor is loaded fine and print some config diagnostics
print(A.config())

# cable parameters
@dataclass
class parameters:
    cm:    float = None
    tempK: float = None
    Vm:    float = None
    rL:    float = None

# parse Allen DB description
# NB. Needs to be adjusted when using a different model
def load_allen_fit(fit):
    with open(fit) as fd:
        fit = json.load(fd)

    param = defaultdict(parameters)
    mechs = defaultdict(dict)
    for block in fit['genome']:
        mech   = block['mechanism'] or 'pas'
        region = block['section']
        name   = block['name']
        value  = float(block['value'])
        if name.endswith('_' + mech):
            name = name[:-(len(mech) + 1)]
        else:
            if mech == "pas":
                # transform names and values
                if name == 'cm':
                    # scaling factor NEURON -> Arbor
                    param[region].cm = value/100.0
                elif name == 'Ra':
                    param[region].rL = value
                elif name == 'Vm':
                    param[region].Vm = value
                elif name == 'celsius':
                    param[region].tempK = value + 273.15
                else:
                    raise Exception(f"Unknown key: {name}")
                continue
            else:
                raise Exception(f"Illegal combination {mech} {name}")
        mechs[(region, mech)][name] = value

    param = [(r, vs) for r, vs in param.items()]
    mechs = [(r, m, vs) for (r, m), vs in mechs.items()]

    default = parameters(None, # not set in example file
                         float(fit['conditions'][0]['celsius']) + 273.15,
                         float(fit['conditions'][0]['v_init']),
                         float(fit['passive'][0]['ra']))

    erev = []
    for kv in fit['conditions'][0]['erev']:
        region = kv['section']
        for k, v in kv.items():
            if k == 'section':
                continue
            ion = k[1:]
            erev.append((region, ion, float(v)))

    return default, param, erev, mechs


class recipe(A.recipe):
    def __init__(self, swc, fit, epas_is_param=True):
        # Need to call this for proper initialisation
        A.recipe.__init__(self)
        # (1) Load the cell morphology.
        self.morphology = A.load_swc_neuron(swc)
        # (2) Label the region tags found in the swc with the names used in the parameter fit file.
        # In addition, label the midpoint of the soma.
        self.labels = A.label_dict({'soma': '(tag 1)',
                                    'axon': '(tag 2)',
                                    'dend': '(tag 3)',
                                    'apic': '(tag 4)',
                                    'center': '(location 0 0.5)'})

        dflt, regions, ions, mechanisms = load_allen_fit(fit)

        # (3) Instantiate an empty decor.
        self.decor = A.decor()
        # (4) assign global electro-physical parameters
        self.decor.set_property(tempK=dflt.tempK, Vm=dflt.Vm, cm=dflt.cm, rL=dflt.rL)
        # (5) override regional electro-physical parameters
        for region, vs in regions:
            self.decor.paint(f'"{region}"', tempK=vs.tempK, Vm=vs.Vm, cm=vs.cm, rL=vs.rL)
        # (6) set reversal potentials
        for region, ion, e in ions:
            self.decor.paint(f'"{region}"', ion_name=ion, rev_pot=e)
        self.decor.set_ion('ca', int_con=5e-5, ext_con=2.0, method=A.mechanism('nernst/x=ca'))
        # (7) assign ion dynamics
        for region, mech, values in mechanisms:
            nm = mech
            vs = {}
            sp = '/'
            for k, v in values.items():
                # pas::e became a global between .5 and .6, toggle epas_is_param
                if epas_is_param and mech == 'pas' and k == 'e':
                    nm = f'{nm}{sp}{k}={v}'
                    sp = ','
                else:
                    vs[k] = v
            self.decor.paint(f'"{region}"', A.density(A.mechanism(nm, vs)))
        # (8) attach stimulus and spike detector
        self.decor.place('"center"', A.iclamp(200, 1000, 0.15), 'ic')
        self.decor.place('"center"', A.spike_detector(-40), 'sd')
        # (9) discretisation strategy: max compartment length
        self.decor.discretization(A.cv_policy_max_extent(20))

        # (10) Create cell
        self.cell = A.cable_cell(self.morphology, self.labels, self.decor)

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, _):
        return self.cell

    def probes(self, _):
        # (11) Set the probe
        return [A.cable_probe_membrane_voltage('"center"')]

    def global_properties(self, kind):
        props = A.neuron_cable_properties()
        props.catalogue = A.allen_catalogue()
        props.catalogue.extend(A.default_catalogue(), '')
        # (12) Install the Allen mechanism catalogue.
        return props


# (13) Setup and run simulation
ctx = A.context()
mdl = recipe('single_cell_allen.swc', 'single_cell_allen_fit.json')
ddc = A.partition_load_balance(mdl, ctx)
sim = A.simulation(mdl, ddc, ctx)
hdl = sim.sample((0, 0), A.regular_schedule(0.005))
sim.record(A.spike_recording.all)
sim.run(tfinal=1400, dt=0.005)

# (14) Load and scale reference
reference = 1000.0*pd.read_csv('single_cell_allen_neuron_ref.csv')['U/mV'].values[:-1] - 14.0

# (15) Extract data
data, _ = sim.samples(hdl)[0]
ts = data[:, 0]
us = data[:, 1]
spikes = np.array([t for (_, t) in sim.spikes()])

# (16) Plot
fg, ax = plt.subplots()
c_arbor, c_spike, *_ = sns.color_palette()
ax.plot(ts, reference, label='Reference', color='0.5')
ax.plot(ts, us,        label='Arbor', color=c_arbor)
ax.scatter(spikes, -40*np.ones_like(spikes), label='Spikes', color=c_spike, zorder=20)
ax.set_xlim(left=0, right=1400)
ax.legend(loc='upper right')
ax.set_ylabel('Membrane Potential at Soma $(U/mV)$')
ax.set_xlabel('Time $(t/ms)$')
fg.savefig('result.pdf')
