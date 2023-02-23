#!/usr/bin/env python3

from collections import defaultdict
from dataclasses import dataclass
import json
import arbor
import seaborn
import pandas
import matplotlib.pyplot as plt


# (3) A function that parses the Allen parameter fit file into components for an arbor.decor
# NB. Needs to be adjusted when using a different model
def load_allen_fit(fit):
    with open(fit) as fd:
        fit = json.load(fd)

    # cable parameters convenience class
    @dataclass
    class parameters:
        cm: float = None
        tempK: float = None
        Vm: float = None
        rL: float = None

    param = defaultdict(parameters)
    mechs = defaultdict(dict)
    for block in fit["genome"]:
        mech = block["mechanism"] or "pas"
        region = block["section"]
        name = block["name"]
        value = float(block["value"])
        if name.endswith("_" + mech):
            name = name[: -(len(mech) + 1)]
        elif mech == "pas":
            # transform names and values
            if name == "cm":
                # scaling factor NEURON -> Arbor
                param[region].cm = value / 100.0
            elif name == "Ra":
                param[region].rL = value
            elif name == "Vm":
                param[region].Vm = value
            elif name == "celsius":
                param[region].tempK = value + 273.15
            else:
                raise Exception(f"Unknown key: {name}")
            continue
        else:
            raise Exception(f"Illegal combination {mech} {name}")
        mechs[(region, mech)][name] = value

    regs = [(r, vs) for r, vs in param.items()]
    mechs = [(r, m, vs) for (r, m), vs in mechs.items()]

    default = parameters(
        tempK=float(fit["conditions"][0]["celsius"]) + 273.15,
        Vm=float(fit["conditions"][0]["v_init"]),
        rL=float(fit["passive"][0]["ra"]),
    )

    ions = []
    for kv in fit["conditions"][0]["erev"]:
        region = kv["section"]
        for k, v in kv.items():
            if k == "section":
                continue
            ion = k[1:]
            ions.append((region, ion, float(v)))

    return default, regs, ions, mechs, fit["fitting"][0]["junction_potential"]


def make_cell(swc, fit):
    # (1) Load the swc file passed into this function
    morphology = arbor.load_swc_neuron(swc)
    # (2) Label the region tags found in the swc with the names used in the parameter fit file.
    # In addition, label the midpoint of the somarbor.
    labels = arbor.label_dict().add_swc_tags()
    labels["midpoint"] = "(location 0 0.5)"

    # (3) A function that parses the Allen parameter fit file into components for an arbor.decor
    dflt, regions, ions, mechanisms, offset = load_allen_fit(fit)

    # (4) Instantiate an empty decor.
    decor = arbor.decor()
    # (5) assign global electro-physiology parameters
    decor.set_property(tempK=dflt.tempK, Vm=dflt.Vm, cm=dflt.cm, rL=dflt.rL)
    # (6) override regional electro-physiology parameters
    for region, vs in regions:
        decor.paint(f'"{region}"', tempK=vs.tempK, Vm=vs.Vm, cm=vs.cm, rL=vs.rL)
    # (7) set reversal potentials
    for region, ion, e in ions:
        decor.paint(f'"{region}"', ion_name=ion, rev_pot=e)
    decor.set_ion(
        "ca", int_con=5e-5, ext_con=2.0, method=arbor.mechanism("nernst/x=ca")
    )
    # (8) assign ion dynamics
    for region, mech, values in mechanisms:
        nm = mech
        vs = {}
        sp = "/"
        for k, v in values.items():
            if mech == "pas" and k == "e":
                nm = f"{nm}{sp}{k}={v}"
                sp = ","
            else:
                vs[k] = v
        decor.paint(f'"{region}"', arbor.density(arbor.mechanism(nm, vs)))
    # (9) attach stimulus and detector
    decor.place('"midpoint"', arbor.iclamp(200, 1000, 0.15), "ic")
    decor.place('"midpoint"', arbor.threshold_detector(-40), "sd")
    # (10) discretisation strategy: max compartment length
    decor.discretization(arbor.cv_policy_max_extent(20))

    # (11) Create cell
    return arbor.cable_cell(morphology, decor, labels), offset


# (12) Create cell, model
cell, offset = make_cell("single_cell_allen.swc", "single_cell_allen_fit.json")
model = arbor.single_cell_model(cell)

# (13) Set the probe
model.probe("voltage", '"midpoint"', frequency=200)

# (14) Install the Allen mechanism catalogue.
model.properties.catalogue.extend(arbor.allen_catalogue(), "")

# (15) Run simulation
model.run(tfinal=1400, dt=0.005)

# (16) Load and scale reference
reference = (
    1000.0 * pandas.read_csv("single_cell_allen_neuron_ref.csv")["U/mV"].values[:-1]
    + offset
)

# (17) Plot
df_list = []
df_list.append(
    pandas.DataFrame(
        {
            "t/ms": model.traces[0].time,
            "U/mV": model.traces[0].value,
            "Simulator": "Arbor",
        }
    )
)
df_list.append(
    pandas.DataFrame(
        {"t/ms": model.traces[0].time, "U/mV": reference, "Simulator": "Neuron"}
    )
)
df = pandas.concat(df_list, ignore_index=True)
seaborn.relplot(
    data=df, kind="line", x="t/ms", y="U/mV", hue="Simulator", errorbar=None
)
plt.scatter(
    model.spikes, [-40] * len(model.spikes), color=seaborn.color_palette()[2], zorder=20
)
plt.bar(
    200,
    max(reference) - min(reference),
    1000,
    min(reference),
    align="edge",
    label="Stimulus",
    color="0.9",
)
plt.savefig("single_cell_allen_result.svg")
