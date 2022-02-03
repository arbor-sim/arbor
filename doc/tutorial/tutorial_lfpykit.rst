.. _tutorial_lfpykit:

Extracellular signals
=====================

This example takes elements from other tutorials
(:ref:`A simple single cell model <tutorialsinglecell>`,
:ref:`A detailed single cell recipe <tutorialsinglecellswcrecipe>`)
to create a geometrically
detailed single cell model from an SWC morphology file, and adds predictions of
extracellular potentials using the external `LFPykit <https://lfpykit.readthedocs.io/en/latest>`_ Python library
(https://LFPykit.readthedocs.io, https://github.com/LFPy/LFPykit).
`LFPykit <https://lfpykit.readthedocs.io/en/latest>`_ provides a few different classes facilitating
calculations of extracellular potentials and related electroencephalography (EEG)
and magnetoencephalography (MEG) signals from geometrically detailed neuron models under various assumptions.
These are signals that mainly stem from transmembrane currents.


.. _tutorial_lfpykit-linesource:

The line source approximation
-----------------------------

First, let's describe how one can compute extracellular potentials from transmembrane currents of a number of segments,
assuming that each segment can be treated as an equivalent line current source using a formalism invented
by Gary R. Holt and Christof Koch [1]_.
The implementation used in this tutorial uses :class:`lfpykit.LineSourcePotential`
(`lfpykit.LineSourcePotential <https://lfpykit.readthedocs.io/en/latest/#class-linesourcepotential>`_).
This class conveniently defines a 2D linear response matrix
:math:`\mathbf{M}` between transmembrane current array
:math:`\mathbf{I}` (nA) of a neuron model and the
corresponding extracellular electric potential in different extracellular locations
:math:`\mathbf{V}_{e}` (mV) so

.. math:: \mathbf{V}_{e} = \mathbf{M} \mathbf{I}

The transmembrane current :math:`\mathbf{I}` is an array of shape (# segments, # timesteps)
with unit (nA), and each row indexed by :math:`j` of
:math:`\mathbf{V}_{e}` represents the electric potential at each
measurement site for every time step.

The elements of :math:`\mathbf{M}` are computed as

.. math:: M_{ji} = \frac{1}{ 4 \pi \sigma L_i } \log
    \left|
    \frac{\sqrt{h_{ji}^2+r_{ji}^2}-h_{ji}
           }{
           \sqrt{l_{ji}^2+r_{ji}^2}-l_{ji}}
    \right|


Here, segments are indexed by :math:`i`,
segment length is denoted :math:`L_i`, perpendicular distance from the
electrode point contact to the axis of the line segment is denoted
:math:`r_{ji}`, longitudinal distance measured from the start of the
segment is denoted :math:`h_{ji}` and longitudinal distance from the other
end of the segment is denoted :math:`l_{ji}= L_i + h_{ji}`.

.. Note::

    **Assumptions:**

    1. The extracellular conductivity :math:`\sigma` is infinite, homogeneous, frequency independent (linear) and isotropic
    2. Each segment is treated as a straight line source with homogeneous current density between its start and end point coordinate.
       Although Arbor allows segments to be defined as conical frusta with varying radius, we shall assume that any variation in
       radius is small relative to overall segment length.
    3. Each measurement site :math:`\mathbf{r}_j = (x_j, y_j, z_j)` is treated as a point
    4. The minimum distance to a line source is set equal to the average segment radius to avoid singularities.


.. _tutorial_lfpykit-model:

The model
---------

In this tutorial, the neuron model itself is kept deliberately simple with only
passive (leaky) membrane dynamics, and it receives sinusoid synaptic current
input in one arbitrary chosen control volume (CV).

.. Note::

   **Concepts covered in this tutorial:**

   1. Building a morphology from an SWC file.
   2. Recording of transmembrane currents using :class:`arbor.cable_probe_total_current_cell`
   3. Recording of stimulus currents using :class:`arbor.cable_probe_stimulus_current_cell`
   4. Calling the :class:`arbor.place_pwlin` API
   5. Map recorded transmembrane currents to extracellular potentials using `LFPykit <https://lfpykit.readthedocs.io/en/latest>`_



First we import some required modules:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 14-17


Define a very basic ``Recipe`` class, holding a cell and three probes (voltage, stimulus current and total current):

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 22-54


Load morphology on ``SWC`` file format (interpreted according to :ref:`Arbor's specifications <morph-formats>`).
Similar to the tutorial :ref:`"A simple single cell model" <tutorialsinglecellswc>`
we parse a morphology file ``single_cell_detailed.swc`` defined as

.. literalinclude:: ../../python/example/single_cell_detailed.swc
   :language: python


as an argument to the simulation script:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 57-66

Define various attributes (:class:`arbor.label_dict`, :class:`arbor.decor`) for
the cell model.
Finally, we define a discretization policy (:class:`arbor.cv_policy_fixed_per_branch`):

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 68-94

Set sinusoid current clamp as stimuli using :class:`arbor.iclamp` at a location at a
relative distance :math:`1/6` from the root to the tip of the dendritic branch with tag ``4``.
The location is marked in the result :ref`figure <tutorial_lfpykit-illustration>`:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 96-103

Create :class:`arbor.place_pwlin` instance:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 106


Define :class:`arbor.cable_cell`, :class:`Recipe`, :class:`arbor.context` etc. and execute model for a few hundred ms,
sampling the different signals every 1 ms:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 108-126


Extract recorded membrane voltages, electrode and transmembrane currents.
Note that membrane voltages at branch points and intersections between CVs are dropped as
we only illustrate membrane voltages of segments with finite lengths.

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 128-143


Finally we sum the stimulation and transmembrane currents, allowing the stimuli to mimic a synapse
current embedded in the membrane itself rather than an intracellular electrode current:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 151


.. _tutorial_lfpykit-lfpykit:

Compute extracellular potentials
--------------------------------

Here we utilize the `LFPykit <https://lfpykit.readthedocs.io/en/latest>`_ library to map
transmembrane currents recorded during the
simulation to extracellular potentials in vicinity to the cell.
We shall account for every segment in each CV using the so-called line-source approximation described :ref:`above <tutorial_lfpykit-linesource>`.

First we define a couple of inherited classes to interface `LFPykit <https://lfpykit.readthedocs.io/en/latest>`_
(as this library is not solely written for Arbor).
Starting with a class inherited from :class:`lfpykit.CellGeometry`
(`lfpykit.CellGeometry <https://lfpykit.readthedocs.io/en/latest/#class-cellgeometry>`_):

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 161-194

Then, a class inherited from :class:`lfpykit.LineSourcePotential`
(`lfpykit.LineSourcePotential <https://lfpykit.readthedocs.io/en/latest/#class-linesourcepotential>`_).
Other use cases may inherit from any other parent class defined in :class:`lfpykit.models`
(`lfpykit.models <https://lfpykit.readthedocs.io/en/latest/#module-lfpykit.models>`_) in a similar manner:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 197-244


With these two classes one may then compute extracellular potentials from transmembrane
currents in space with a few lines of code:

.. literalinclude:: ../../python/example/single_cell_extracellular_potentials.py
   :language: python
   :lines: 247-270


.. _tutorial_lfpykit-illustration:

The result
----------

The visualization below of simulation results shows the cellular geometry and a contour plot
of the extracellular potential (``V_e``) in a plane.
Each part (CV) of the cell is shown with some color coding for the membrane potential (``V_m``).
The stimulus location is denoted by the black marker.


.. figure:: tutorial_lfpykit.svg
    :width: 1600
    :align: center


.. Note::

    The spatial discretization is here deliberately coarse with only 3 CVs per branch.
    Hence the branch receiving input about 1/6 of the way from its root
    (from ``decor.place('(location 4 0.16667)', iclamp, '"iclamp"')``) is treated as 3 separate
    line sources with inhomogeneous current density per unit length. This inhomogeneity
    is due to the fact that the total transmembrane current per CV may
    be distributed across multiple segments with varying surface area. The transmembrane
    current is assumed to be constant per unit length per segment.


.. _tutorial_lfpykit-code:

The full code
-------------
You can find the full code of the example at ``python/examples/single_cell_extracellular_potentials.py``.


References
----------
.. [1] Holt, G.R., Koch, C. Electrical Interactions via the Extracellular Potential Near Cell Bodies.
  J Comput Neurosci 6, 169–184 (1999). https://doi.org/10.1023/A:1008832702585
