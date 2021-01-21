.. _tutorialsinglecellallen:

A single cell model from the Allen Brain Atlas
==============================================

In this tutorial we'll see how we can take a model from the `Allen Brain Atlas <https://brain-map.org/>`_
and run it with Arbor.

.. Note::

   **Concepts covered in this example:**

   1. Take a model from an online atlas.
   1. Load a morphology from an ``swc`` file.
   2. Load a parameter fit file and apply it to a :class:`arbor.decor`.
   4. Building a :class:`arbor.cable_cell` object.
   5. Building a :class:`arbor.single_cell_model` object.
   6. Running a simulation and visualising the results.

Obtaining the model
-------------------

We need a model for which morphological data is available. We'll take
`a cell from the mouse's visual cortex <http://celltypes.brain-map.org/experiment/electrophysiology/488683425>`_.
As the ``README`` in the "Biophysical - all active" model (zip) file explains, the data set is created for use with the Allen SDK and the
Neuron simulator. Instructions about running the model with the
Allen SDK and Neuron `can be found here <https://allensdk.readthedocs.io/en/latest/biophysical_models.html>`_.
For your convenience, this tutorial comes with a reference trace pre-generated.

In the "Biophysical - all active" model (zip) file you'll find:

* a directory with mechanisms (modfiles). For your convenience, Arbor already comes with the
  :func:`arbor.allen_catalogue`, containing mechanisms used by the Allen Brain Atlas.
* an ``swc`` file describing the morphology of the cell.
* ``fit_parameters.json`` describing a set of optimized mechanism parameters. This information will need to be
  loaded into the Arbor simulation.

We will replicate the "Sweep 35" experiment, which applies a current of 150 nA for a duration of 1 s.

The morphology
--------------

:ref:`In an earlier tutorial <tutorialsinglecellswc-cell>` we've seen how an ``swc`` file can be loaded **(1)**
and how the labels can be set **(2)**:

.. code-block:: python

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

Step **(1)** loads the ``swc`` file using :func:`arbor.load_swc_allen`. Since the ``swc`` specification is informal, a few different interpretations exist, and we use the appropriate one. The interpretations are described ::ref`here <morph-formats>`.

Step **(2)** sets the labels to the defaults of the ``swc``
`specification <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_,
plus a label for the midpoint of the soma. (You can verify in the ``swc`` file, the first branch is the soma.)

The decor
---------

The most complicated part is transferring the values for the appropriate parameters in parameter fit file to an
:class:`arbor.decor`. The file file is a ``json`` file, which is fortunate; Python comes with a ``json`` package
in its standard library. The `passive` and `conditions` block contains cell-wide defaults, while the `genome`
section contains the parameters for all the mechanism properties. In certain cases, parameters names include the
mechanism name, so some processing needs to take place.

Step **(3)** shows the precise steps needed to load the fit parameter file into a list of global properties,
region specific properties, reversal potentials, and mechanism parameters.

.. code-block:: Python

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


Step **(3)** creates an empty :class:`arbor.decor`.

Step **(4)** assigns global (cell-wide) properties using :func:`arbor.decor.set_property`. In addition, initial
internal and external calcium concentrations are set, and configured to be mediated by the Nernst equation.

.. note::
    Setting the calcium reversal potential to be mediated by the Nernst equation has to be done manually, in order to mirror
    `an implicit Neuron behavior <https://neuron.yale.edu/neuron/static/new_doc/modelspec/programmatic/ions.html>`_,
    for which the fit parameters were obtained. This behavior can be stated as the following rule:

    If the internal or external concentration of an ion is written, and its reversal potential is read but not
    written, then the nernst equation is used continuously during the simulation to update the reversal potential of
    the ion according to the nernst equation

Step **(5)** overrides the global properties for all *regions* for which the fit parameters file specifies adapted
values. Regional properties are :func:`painted `arbor.paint`, and are painted over the defaults.

Step **(6)** sets the regional reversal potentials.

Step **(7)** assigns the regional mechanisms.

Now that the electrodynamics are all set up, let's move on to the experimental setup.

Step **(8)** configures the :class:`stimulus <arbor.iclamp>` of 150 nA for a duration of 1 s, starting after 200 ms
of the start of the simulation. We'll also install a :class:`arbor.spike_detector` that triggers at -40 mV. (The
location is usually the soma, as is confirmed by coordinates found in the experimental dataset at
``488683423.nwb/general/intracellular_ephys/Electrode 1/location``)

Step **(9)** specifies a maximum :term:`control volume` length of 20 μm.

The model
---------

.. code-block:: python

  # (10) Create cell, model
  cell = arbor.cable_cell(morphology, labels, decor)
  model = arbor.single_cell_model(cell)

  # (11) Set the probe
  model.probe('voltage', '"midpoint"', frequency=200000)

  # (12) Install the Allen mechanism catalogue.
  model.catalogue.extend(arbor.allen_catalogue(), "")

  # (13) Run simulation
  model.run(tfinal=1400, dt=0.005)

Step **(10)** creates the :class:`arbor.cable_cell` and :class:`arbor.single_cell_model`.

Step **(11)** shows how to install a probe to the ``"midpoint"``, with a sampling frequency of 200 kHz.

Step **(12)** installs the :class:`arbor.allen_catalogue`, thereby making its mechanisms available to the definitions added to the decor.

Step **(13)** starts the simulation for a duration of 1.4 s and a timestep of 5 ms.

The result
----------

Let's look at the result! In step **(14)** we first load the reference generated with Neuron and the AllenSDK.
Then, we place Arbor's output, accessible after the simulation ran through
:class:`arbor.single_cell_model.traces`. Then, we plot them, together with the :class:`arbor.single_cell_model.spikes`.

.. code-block:: python

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

.. figure:: single_cell_allen_result.svg
    :width: 400
    :align: center

    Plot of experiment 35 of the Allen model, compared to the reference generated by the AllenSDK.

.. note::

  The careful observer notices that this trace does not match the experimental data shown on the Allen website
  (or in the ``488683423.nwb`` file). Sweep 35 clearly has 5 spikes, not 4. That is because in the Allen SDK,
  the axon in the ``swc`` file is replaced with a stub, see
  `here <https://www.biorxiv.org/content/10.1101/2020.04.09.030239v1.full>`_ and `here <https://github.com/AllenInstitute/AllenSDK/issues/1683>`_.
  However, that adapted morphology is not exportable back to a modified ``swc`` file. When we tried to mimic
  the procedure, we did not obtain the experimental trace.

  Therefore, we used the unmodified morphology in Arbor *and* the Neuron reference (by commenting out the
  changes the Allen SDK makes to the morphology) in order to make a 1:1 comparison possible.

The full code
-------------

You can find the source code for this example in full at ``python/examples/single_cell_allen.py``.
