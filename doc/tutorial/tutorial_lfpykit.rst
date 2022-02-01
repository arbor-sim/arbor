.. _tutorial_lfpykit:

Extracellular signals
=====================

This example takes different elements from the above example tutorial(s)
:ref:`A simple single cell recipe <tutorialsinglecell>` above to create a geometrically
complex single cell model from an SWC morphology file, and adds predictions of
extracellular potentials using the external LFPykit Python library
(https://LFPykit.readthedocs.io, https://github.com/LFPy/LFPykit).
This tutorial goes through the required steps in more detail.

See also the tutorial :ref:`A single cell <tutorialsinglecellswcrecipe>` for more
in-depth explanation of recipes.

In this tutorial, the neuron model itself is kept deliberately simple with only
passive (leaky) membrane dynamics, and it receives sinusoid synaptic current
input in one arbitrary chosen control volume (CV).

.. Note::

   **Concepts covered in this example:**

   1. Building a morphology from an SWC file.
   2. Recording of transmembrane currents using :class:`arbor.cable_probe_total_current_cell`
   3. Recording of stimulus currents using :class:`arbor.cable_probe_stimulus_current_cell`
   4. Calling the :class:`arbor.place_pwlin` API
   5. Map recorded transmembrane currents to extracellular potentials using LFPykit


.. _tutorial_lfpykit-model:

The model
*********

Import modules:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 14-21


Define ``Recipe`` class:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 24-56


Load morphology on ``SWC`` file format from the command line:

.. code-block:: python

   # define morphology needed for ``arbor.place_pwlin`` and ``arbor.cable_cell`` below
   morphology = arbor.load_swc_arbor('single_cell_detailed.swc')


Define various attributes for the cell model as well as stimuli:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 70-104

Create :class:`arbor.place_pwlin` instance:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 107


Define ``cell``, ``recipe``, ``context`` and execute model for a few hundred ms:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 109-127


Extract recorded membrane voltages, electrode and transmembrane currents:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 129-153


.. _tutorial_lfpykit-lfpykit:

Compute extracellular potentials
********************************

Here we utilize the LFPykit library to map transmembrane currents recorded during the
simulation to extracellular potential in vicinity to the cell.
We shall account for every segment in each CV using the so-called line-source approximation.

First we define a couple of classes to interface LFPykit (as this library is not solely written for Arbor).
Starting with a class inherited from :class:`lfpykit.CellGeometry`:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 163-196

Then, a class inherited from :class:`lfpykit.LineSourcePotential`.
Other use cases may inherit from any other parent class defined in :class:`lfpykit.models`
(https://lfpykit.readthedocs.io/en/latest/#module-lfpykit.models) in a similar manner:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 199-246


With these two classes one may then compute extracellular potentials from transmembrane currents in space:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 249-273


.. _tutorial_lfpykit-illustration:

The result
************

The visualization below of simulation results shows both the cellular geometry and a countour plot
of the extracellular potential (`V_e`) in a plane.
Each part (CV) of the cell is shown with some color coding for the membrane potential (`V_m`).



.. figure:: tutoriallfpykit.svg
    :width: 1600
    :align: center

The full code
*************
You can find the full code of the example at ``python/examples/single_cell_extracellular_potentials.py``.
