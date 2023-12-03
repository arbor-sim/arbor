#!/usr/bin/env python3

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import json
import arbor as A
from arbor import units as U
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

here = Path(__file__).parent


# (3) A function that parses the Allen parameter fit file into components for an A.decor
# NB. Needs to be adjusted when using a different model
def load_allen_fit(fit):
    with open(fit) as fd:
        fit = json.load(fd)

    # cable parameters convenience class
    @dataclass
    class parameters:
        cm: Optional[float] = None
        tempK: Optional[float] = None
        Vm: Optional[float] = None
        rL: Optional[float] = None

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


def make_cell(base, swc, fit):
    # (1) Load the swc file passed into this function
    morphology = A.load_swc_neuron(base / swc)

    # (2) Label the region tags found in the swc with the names used in the parameter fit file.
    # In addition, label the midpoint of the somA.
    labels = A.label_dict().add_swc_tags()
    labels["midpoint"] = "(location 0 0.5)"

    # (3) A function that parses the Allen parameter fit file into components for an A.decor
    dflt, regions, ions, mechanisms, offset = load_allen_fit(base / fit)

    # (4) Instantiate an empty decor.
    decor = A.decor()

    # (5) assign global electro-physiology parameters
    decor.set_property(
        tempK=dflt.tempK * U.Kelvin,
        Vm=dflt.Vm * U.mV,
        cm=dflt.cm * U.F / U.m2,
        rL=dflt.rL * U.Ohm * U.cm,
    )

    # (6) override regional electro-physiology parameters
    for region, vs in regions:
        decor.paint(
            f'"{region}"',
            tempK=vs.tempK * U.Kelvin,
            Vm=vs.Vm * U.Vm,
            cm=vs.cm * U.F / U.m2,
            rL=vs.rL * U.Ohm * U.cm,
        )

    # (7) set reversal potentials
    for region, ion, e in ions:
        decor.paint(f'"{region}"', ion=ion, rev_pot=e)
    decor.set_ion("ca", int_con=5e-5 * U.mM, ext_con=2.0 * U.mM, method="nernst/x=ca")

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
        decor.paint(f'"{region}"', A.density(A.mechanism(nm, vs)))

    # (9) attach stimulus and detector
    decor.place('"midpoint"', A.iclamp(0.2 * U.s, 1 * U.s, 150 * U.pA), "ic")
    decor.place('"midpoint"', A.threshold_detector(-40 * U.mV), "sd")

    # (10) discretisation strategy: max compartment length
    decor.discretization(A.cv_policy_max_extent(20))

    # (11) Create cell
    return A.cable_cell(morphology, decor, labels), offset


# (12) Create cell, model
cell, offset = make_cell(here, "single_cell_allen.swc", "single_cell_allen_fit.json")
model = A.single_cell_model(cell)

# (13) Set the probe
model.probe("voltage", '"midpoint"', "Um", frequency=1 / (5 * U.us))

# (14) Install the Allen mechanism catalogue.
model.properties.catalogue.extend(A.allen_catalogue(), "")

# (15) Run simulation
model.run(tfinal=1.4 * U.s, dt=5 * U.us)

# (16) Load and scale reference
reference = (
    1e3 * pd.read_csv(here / "single_cell_allen_neuron_ref.csv")["U/mV"] + offset
)

# (17) Plot
df = pd.concat(
    [
        pd.DataFrame(
            {
                "Simulator": "Arbor",
                "t/ms": model.traces[0].time,
                "U/mV": model.traces[0].value,
            }
        ),
        pd.DataFrame(
            {
                "Simulator": "Neuron",
                "t/ms": model.traces[0].time,
                "U/mV": reference.values,
            }
        ),
    ],
    ignore_index=True,
)

sns.relplot(data=df, kind="line", x="t/ms", y="U/mV", hue="Simulator", errorbar=None)
plt.scatter(
    model.spikes, [-40] * len(model.spikes), color=sns.color_palette()[2], zorder=20
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
plt.savefig("single_cell_allen_result.pdf")
