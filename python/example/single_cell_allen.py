#!/usr/bin/env python3
import arbor, pandas, seaborn, matplotlib.pyplot as plt

# (1) Load the cell morphology.
morphology = arbor.load_swc_allen('single_cell_allen.swc', no_gaps=False)
# (2) Label the region tags found in the swc with the names used in the parameter fit file.
# In addition, label the midpoint of the soma.
labels = arbor.label_dict({
    'soma': '(tag 1)',
    'axon': '(tag 2)',
    'dend': '(tag 3)',
    'apic': '(tag 4)',
    'midpoint': '(location 0 0.5)'})

# (3) A function that parses the Allen parameter fit file into components for an arbor.decor
def load_allen_fit(fit):
    from collections import defaultdict
    import json
    from dataclasses import dataclass

    @dataclass
    class parameters:
        cm:    float = None
        tempK: float = None
        Vm:    float = None
        rL:    float = None

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
        if mech == 'pas':
            mech = 'pas'
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

    pot_offset = fit['fitting'][0]['junction_potential']

    return default, param, erev, mechs, pot_offset

defaults, regions, ions, mechanisms, pot_offset = load_allen_fit('single_cell_allen_fit.json')

# (3) Instantiate an empty decor.
decor = arbor.decor()

# (4) assign global electro-physical parameters
decor.set_property(tempK=defaults.tempK, Vm=defaults.Vm,
                    cm=defaults.cm, rL=defaults.rL)
decor.set_ion('ca', int_con=5e-5, ext_con=2.0, method=arbor.mechanism('nernst/x=ca'))
# (5) override regional electro-physical parameters
for region, vs in regions:
    decor.paint('"'+region+'"', tempK=vs.tempK, Vm=vs.Vm, cm=vs.cm, rL=vs.rL)
# (6) set reversal potentials
for region, ion, e in ions:
    decor.paint('"'+region+'"', ion, rev_pot=e)
# (7) assign ion dynamics
for region, mech, values in mechanisms:
    decor.paint('"'+region+'"', arbor.mechanism(mech, values))

# (8) attach stimulus and spike detector
decor.place('"midpoint"', arbor.iclamp(200, 1000, 0.15))
decor.place('"midpoint"', arbor.spike_detector(-40))

# (9) discretisation strategy: max compartment length
decor.discretization(arbor.cv_policy_max_extent(20))

# (10) Create cell, model
cell = arbor.cable_cell(morphology, labels, decor)
model = arbor.single_cell_model(cell)

# (11) Set the probe
model.probe('voltage', '"midpoint"', frequency=200000)

# (12) Install the Allen mechanism catalogue.
model.catalogue.extend(arbor.allen_catalogue(), "")

# (13) Run simulation
model.run(tfinal=1400, dt=0.005)

# (14) Load reference data and plot results.
reference = pandas.read_csv('single_cell_allen_neuron_ref.csv')

df = pandas.DataFrame()
for t in model.traces:
     # need to shift by junction potential, see allen db
    df=df.append(pandas.DataFrame({'t/ms': t.time, 'U/mV': [i-pot_offset for i in t.value], 'Variable': t.variable, 'Simulator': 'Arbor'}))
# neuron outputs V instead of mV
df=df.append(pandas.DataFrame({'t/ms': reference['t/ms'], 'U/mV': 1000.0*reference['U/mV'], 'Variable': 'voltage', 'Simulator':'Neuron'}))

seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Simulator",col="Variable",ci=None)

plt.scatter(model.spikes, [-40]*len(model.spikes), color=seaborn.color_palette()[2], zorder=20)
plt.savefig('single_cell_allen_result.svg')
