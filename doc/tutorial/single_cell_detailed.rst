.. _tutorialsinglecellswc:

A detailed single cell model
============================

We can expand on the :ref:`single segment cell example <tutorialsinglecell>` to create a more
complex single cell model, and go through the process in more detail.

.. Note::

   **Concepts covered in this example:**

   1. Building a morphology from a :class:`arbor.segment_tree`.
   2. Building a morphology from an SWC file.
   3. Writing and visualizing region and locset expressions.
   4. Building a decor.
   5. Discretising the morphology.
   6. Setting and overriding model and cell parameters.
   7. Running a simulation and visualising the results using a :class:`arbor.single_cell_model`.

We start by building the cell. This will be a :ref:`cable cell <cablecell>` with complex
geometry and dynamics which can be constructed from 3 components:

1. A **morphology** defining the geometry of the cell.
2. A **label dictionary** storing labelled expressions which define regions and locations of
   interest on the cell.
3. A **decor** defining various properties and dynamics on these  regions and locations.
   The decor also includes hints about how the cell is to be modelled under the hood, by
   splitting it into discrete control volumes (CV).

Next, we construct a :class:`arbor.single_cell_model`. This model takes care of a lot of details
behind the scenes: it sets up a recipe (more on recipes :ref:`here <modelrecipe>`), creates
a simulation object, manages the hardware etc. These details become more important when modelling
a network of cells, but can be abstracted away when working with single cell networks.

The single cell model has 4 main functions:

1. It holds the **global properties** of the model
2. It registers **probes** on specific locations on the cell to measure the voltage.
3. It **runs** the simulation.
4. It collects **spikes** from spike detectors and voltage **traces** from registered probes.

.. _tutorialsinglecellswc-cell:

The cell
********

Before creating the actual cell object, we have to create its components.

The morphology
^^^^^^^^^^^^^^^
We begin by constructing the following morphology:

.. figure:: ../gen-images/tutorial_morph.svg
   :width: 400
   :align: center

This can be done by manually building a segment tree:

.. code-block:: python

    import arbor
    from arbor import mpoint
    from arbor import mnpos

    # Define the morphology by manually building a segment tree

    tree = arbor.segment_tree()

    # Start with segment 0: a cylindrical soma with tag 1
    tree.append(mnpos, mpoint(0.0, 0.0, 0.0, 2.0), mpoint( 40.0, 0.0, 0.0, 2.0), tag=1)
    # Construct the first section of the dendritic tree with tag 3,
    # comprised of segments 1 and 2, attached to soma segment 0.
    tree.append(0,     mpoint(40.0, 0.0, 0.0, 0.8), mpoint( 80.0,  0.0, 0.0, 0.8), tag=3)
    tree.append(1,     mpoint(80.0, 0.0, 0.0, 0.8), mpoint(120.0, -5.0, 0.0, 0.8), tag=3)
    # Construct the rest of the dendritic tree: segments 3, 4 and 5.
    tree.append(2,     mpoint(120.0, -5.0, 0.0, 0.8), mpoint(200.0,  40.0, 0.0, 0.4), tag=3)
    tree.append(3,     mpoint(200.0, 40.0, 0.0, 0.4), mpoint(260.0,  60.0, 0.0, 0.2), tag=3)
    tree.append(2,     mpoint(120.0, -5.0, 0.0, 0.5), mpoint(190.0, -30.0, 0.0, 0.5), tag=3)
    # Construct a special region of the tree made of segments 6, 7, and 8
    # differentiated from the rest of the tree using tag 4.
    tree.append(5,     mpoint(190.0, -30.0, 0.0, 0.5), mpoint(240.0, -70.0, 0.0, 0.2), tag=4)
    tree.append(5,     mpoint(190.0, -30.0, 0.0, 0.5), mpoint(230.0, -10.0, 0.0, 0.2), tag=4)
    tree.append(7,     mpoint(230.0, -10.0, 0.0, 0.2), mpoint(360.0, -20.0, 0.0, 0.2), tag=4)
    # Construct segments 9 and 10 that make up the axon with tag 2.
    # Segment 9 is at the root, where its proximal end will be connected to the
    # proximal end of the soma segment.
    tree.append(mnpos, mpoint( 0.0, 0.0, 0.0, 2.0), mpoint(  -70.0, 0.0, 0.0, 0.4), tag=2)
    tree.append(9,     mpoint(-70.0, 0.0, 0.0, 0.4), mpoint(-100.0, 0.0, 0.0, 0.4), tag=2)

    morph = arbor.morphology(tree);

The same morphology can be represented using an SWC file (interpreted according
to :ref:`Arbor's specifications <morph-formats>`). We can save the following in
``single_cell_detailed.swc``.

.. code-block:: python

   # id,  tag,      x,      y,      z,      r,    parent
       1     1     0.0     0.0     0.0     2.0        -1  # seg0 prox / seg9 prox
       2     1    40.0     0.0     0.0     2.0         1  # seg0 dist
       3     3    40.0     0.0     0.0     0.8         2  # seg1 prox
       4     3    80.0     0.0     0.0     0.8         3  # seg1 dist / seg2 prox
       5     3   120.0    -5.0     0.0     0.8         4  # seg2 dist / seg3 prox
       6     3   200.0    40.0     0.0     0.4         5  # seg3 dist / seg4 prox
       7     3   260.0    60.0     0.0     0.2         6  # seg4 dist
       8     3   120.0    -5.0     0.0     0.5         5  # seg5 prox
       9     3   190.0   -30.0     0.0     0.5         8  # seg5 dist / seg6 prox / seg7 prox
      10     4   240.0   -70.0     0.0     0.2         9  # seg6 dist
      11     4   230.0   -10.0     0.0     0.2         9  # seg7 dist / seg8 prox
      12     4   360.0   -20.0     0.0     0.2        11  # seg8 dist
      13     2   -70.0     0.0     0.0     0.4         1  # seg9 dist / seg10 prox
      14     2  -100.0     0.0     0.0     0.4        13  # seg10 dist

.. Note::

    SWC samples always form a segment with their parent sample. For example,
    sample 3 and sample 2 form a segment which has length = 0.
    We use these zero-length segments to represent an abrupt radius change
    in the morphology, like we see between segment 0 and segment 1 in the above
    morphology diagram.

    More information on SWC loaders can be found :ref:`here <morph-formats>`.

The morphology can then be loaded from ``single_cell_detailed.swc`` in the following way:

.. code-block:: python

    import arbor

    # Read the morphology from an SWC file

    morph = arbor.load_swc_arbor("single_cell_detailed.swc")

The label dictionary
^^^^^^^^^^^^^^^^^^^^

Next, we can define **region** and **location** expressions and give them labels.
The regions and locations are defined using an Arbor-specific DSL, and the labels
can be stored in a :class:`arbor.label_dict`.

.. Note::

   The expressions in the label dictionary don't actually refer to any concrete regions
   or locations of the morphology at this point. They are merely descriptions that can be
   applied to any morphology, and depending on its geometry, they will generate different
   regions and locations. However, we will show some figures illustrating the effect of
   applying these expressions to the above morphology, in order to better visualize the
   final cell.

   More information on region and location expressions is available :ref:`here <labels>`.

First, we can define some **regions**, These are continuous parts of the morphology,
They can correspond to full segments or parts of segments. Our morphology already has some
pre-established regions determined by the ``tag`` parameter of the segments. They are
defined as follows:

.. code-block:: python

    # Create a label dictionary

    labels = arbor.label_dict()

    # Add labels for tag 1, 2, 3, 4
    labels['soma'] = '(tag 1)'
    labels['axon'] = '(tag 2)'
    labels['dend'] = '(tag 3)'
    labels['last'] = '(tag 4)'

This will generate the following regions when applied to the previously defined morphology:

.. figure:: ../gen-images/tutorial_tag.svg
  :width: 800
  :align: center

  From left to right: regions "soma", "axon", "dend" and "last"

We can also define a region that represents the whole cell; and to make things a bit more interesting,
a region that includes the parts of the morphology that have a radius greater than 1.5 μm. This is done
in the following way:

.. code-block:: python

    # Add a label for a region that includes the whole morphology
    labels['all'] = '(all)'

    # Add a label for the parts of the morphology with radius greater than 1.5 μm.
    labels['gt_1.5'] = '(radius-gt (region "all") 1.5)'

This will generate the following regions when applied to the previously defined morphology:

.. figure:: ../gen-images/tutorial_all_gt.svg
  :width: 400
  :align: center

  Left: region "all"; right: region "gt_1.5"

By comparing to the original morphology, we can see region "gt_1.5" includes all of segment 0 and part of
segment 9.

Finally, let's define a region that includes two already defined regions: "last" and "gt_1.5". This can
be done as follows:

.. code-block:: python

    # Join regions "last" and "gt_1.5"
    labels['custom'] = '(join (region "last") (region "gt_1.5"))'

This will generate the following region when applied to the previously defined morphology:

.. figure:: ../gen-images/tutorial_custom.svg
  :width: 200
  :align: center

Our label dictionary so far only contains regions. We can also add some **locations**. Let's start
with a location that is the root of the morphology, and the set of locations that represent all the
terminal points of the morphology.

.. code-block:: python

    # Add a labels for the root of the morphology and all the terminal points
    labels['root'] = '(root)'
    labels['terminal'] = '(terminal)'

This will generate the following **locsets** (sets of one or more locations) when applied to the
previously defined morphology:

.. figure:: ../gen-images/tutorial_root_term.svg
  :width: 400
  :align: center

  Left: locset "root"; right: locset "terminal"

To make things more interesting, let's select only the terminal points which belong to the
previously defined "custom" region; and, separately, the terminal points which belong to the
"axon" region:

.. code-block:: python

    # Add a label for the terminal locations in the "custom" region:
    labels['custom_terminal'] = '(restrict (locset "terminal") (region "custom"))'

    # Add a label for the terminal locations in the "axon" region:
    labels['axon_terminal'] = '(restrict (locset "terminal") (region "axon"))'

This will generate the following 2 locsets when applied to the previously defined morphology:

.. figure:: ../gen-images/tutorial_custom_axon_term.svg
  :width: 400
  :align: center

  Left: locset "custom_terminal"; right: locset "axon_terminal"

The decorations
^^^^^^^^^^^^^^^

With the key regions and location expressions identified and labelled, we can start to
define certain features, properties and dynamics on the cell. This is done through a
:class:`arbor.decor` object, which stores a mapping of these "decorations" to certain
region or location expressions.

.. Note::

  Similar to the label dictionary, the decor object is merely a description of how an abstract
  cell should behave, which can then be applied to any morphology, and have a different effect
  depending on the geometry and region/locset expressions.

  More information on decors can be found :ref:`here <cablecell-decoration>`.

The decor object can have default values for properties, which can then be overridden on specific
regions. It is in general better to explicitly set all the default properties of your cell,
to avoid the confusion to having simulator-specific default values. This will therefore be our first
step:

.. code-block:: python

    # Create a decor object
    decor = arbor.decor()

    # Set the default properties
    decor.set_property(Vm =-55, tempK=300, rL=35.4, cm=0.01)
    decor.set_ion('na', int_con=10,   ext_con=140, rev_pot=50, method='nernst/na')
    decor.set_ion('k',  int_con=54.4, ext_con=2.5, rev_pot=-77)

We have set the default initial membrane voltage to -55 mV; the default initial
temperature to 300 K; the default axial resistivity to 35.4 Ω·cm; and the default membrane
capacitance to 0.01 F/m².

We also set the initial properties of the *na* and *k* ions because they will be utilized
by the density mechanisms that we will be adding shortly.
For both ions we set the default initial concentration and external concentration measures in mM;
and we set the default initial reversal potential in mV. For the *na* ion, we additionally indicate
the the progression on the reversal potential during the simulation will be dictated by the
`Nernst equation <https://en.wikipedia.org/wiki/Nernst_equation>`_.

It happens, however, that we want the temperature of the "custom" region defined in the label
dictionary earlier to be colder, and the initial voltage of the "soma" region to be higher.
We can override the default properties by *painting* new values on the relevant regions using
:meth:`arbor.decor.paint`.

.. code-block:: python

    # Override default parameters on certain regions

   decor.paint('"custom"', tempK=270)
   decor.paint('"soma"', Vm=-50)

With the default and initial values taken care of, we now add some density mechanisms. Let's *paint*
a *pas* mechanism everywhere on the cell using the previously defined "all" region; an *hh* mechanism
on the "custom" region; and an *Ih* mechanism on the "dend" region. The *Ih* mechanism is explicitly
constructed in order to change the default values of its 'gbar' parameter.


.. code-block:: python

   # Paint density mechanisms on certain regions

   from arbor import mechanism as mech

   decor.paint('"all"', 'pas')
   decor.paint('"custom"', 'hh')
   decor.paint('"dend"',  mech('Ih', {'gbar': 0.001}))

The decor object is also used to *place* stimuli and spike detectors on the cell using :meth:`arbor.decor.place`.
We place 3 current clamps of 2 nA on the "root" locset defined earlier, starting at time = 10, 30, 50 ms and
lasting 1ms each. As well as spike detectors on the "axon_terminal" locset for voltages above -10 mV.
Every placement gets a label. The labels of detectors and synapses are used to form connection from and to them
in the recipe.

.. code-block:: python

   # Place stimuli and spike detectors on certain locsets

   decor.place('"root"', arbor.iclamp(10, 1, current=2), 'iclamp0')
   decor.place('"root"', arbor.iclamp(30, 1, current=2), 'iclamp1')
   decor.place('"root"', arbor.iclamp(50, 1, current=2), 'iclamp2')
   decor.place('"axon_terminal"', arbor.spike_detector(-10), 'detector')

.. Note::

   The number of individual locations in the ``'axon_terminal'`` locset depends on the underlying morphology and the
   number of axon branches in the morphology. The number of detectors that get added on the cell is equal to the number
   of locations in the locset, and the label ``'detector'`` refers to all of them. If we want to refer to a single
   detector from the group (to form a network connection for example), we need a :py:class:`arbor.selection_policy`.

Finally, there's one last property that impacts the behavior of a model: the discretisation.
Cells in Arbor are simulated as discrete components called control volumes (CV). The size of
a CV has an impact on the accuracy of the results of the simulation. Usually, smaller CVs
are more accurate because they simulate the continuous nature of a neuron more closely.

The user controls the discretisation using a :class:`arbor.cv_policy`. There are a few different policies to
choose from, and they can be composed with one another. In this example, we would like the "soma" region
to be a single CV, and the rest of the morphology to be comprised of CVs with a maximum length of 1 μm:

.. code-block:: python

   # Single CV for the "soma" region
   soma_policy = arbor.cv_policy_single('"soma"')

   # CVs with max length = 1 μm as default
   dflt_policy = arbor.cv_policy_max_extent(1.0)

   # default policy everywhere except the soma
   policy = dflt_policy | soma_policy

   decor.discretization(policy)


The model
*********

We begin by constructing an :class:`arbor.single_cell_model` of the cell we just created.

.. code-block:: python

   # Construct the model

   model = arbor.single_cell_model(cell)

The global properties
^^^^^^^^^^^^^^^^^^^^^

The global properties of a single cell model include:

1. The **mechanism catalogue**: A mechanism catalogue is a collection of density and point
   mechanisms. Arbor has 3 built in mechanism catalogues: default, allen and bbp. The mechanism
   catalogue in the global properties of the model must include the catalogues of all the
   mechanisms painted on the cell decor.

2. The default **parameters**: The initial membrane voltage; the initial temperature; the
   axial resistivity; the membrane capacitance; the ion parameters; and the discretisation
   policy.

.. Note::

   You may notice that the same parameters can be set both at the cell level and at
   the model level. This is intentional. The model parameters apply to all the cells in a model,
   whereas the cell parameters apply only to that specific cell.

   The idea is that the user is able to define a set of global properties for all cells in a model
   which can then be overridden for individual cells, and overridden yet again on certain
   regions of the cells.

   You may now be wondering why this is needed for the `single cell model` where there is only one
   cell by design. You can use this feature to ease moving from simulating a set of single cell models
   to simulating a network of these cells.
   For example, a user may choose to individually test several single cell models before simulating
   their interactions. By using the same global properties for each *model*, and customizing the *cell*
   global properties, it becomes possible to use the cell descriptions of each cell, unchanged, in a
   larger network model.

Earlier in the example we mentioned that it is better to explicitly set all the default properties
of your cell, while that is true, it is better yet to set the default properties of the entire
model:

.. _tutorialsinglecellswc-gprop:

.. code-block:: python

   # Set the model default properties

   model.properties.set_property(Vm =-65, tempK=300, rL=35.4, cm=0.01)
   model.properties.set_ion('na', int_con=10,   ext_con=140, rev_pot=50, method='nernst/na')
   model.properties.set_ion('k',  int_con=54.4, ext_con=2.5, rev_pot=-77)

We set the same properties as we did earlier when we were creating the *decor* of the cell, except
for the initial membrane voltage, which is -65 mV as opposed to -55 mV.

During the decoration step, we also made use of 3 mechanisms: *pas*, *hh* and *Ih*. As it happens,
the *pas* and *hh* mechanisms are in the default Arbor catalogue, whereas the *Ih* mechanism is in
the "allen" catalogue. We can extend the default catalogue as follow:

.. code-block:: python

   # Extend the default catalogue with the allen catalogue.
   # The function takes a second string parameter that can prefix
   # the name of the mechanisms to avoid collisions between catalogues
   # in this case we have no collisions so we use an empty prefix string.

   model.catalogue.extend(arbor.allen_catalogue(), "")

Now all three mechanisms in the *decor* object have been made available to the model.

The probes
^^^^^^^^^^

The model is almost ready for simulation. Except that the only output we would be able to
measure at this point is the spikes from the spike detectors placed in the decor.

The :class:`arbor.single_cell_model` can also measure the voltage on specific locations of the cell.
We can indicate the location we would like to probe using labels from the :class:`label_dict`:

.. code-block:: python

   # Add voltage probes on the "custom_terminal" locset
   # which sample the voltage at 50 kHz

   model.probe('voltage', where='"custom_terminal"', frequency=50)

The simulation
^^^^^^^^^^^^^^

The cell and model descriptions are now complete and we can run the simulation:

.. code-block:: python

   # Run the simulation for 100 ms, with a dt of 0.025 ms
   model.run(tfinal=100, dt=0.025)

The results
^^^^^^^^^^^

Finally we move on to the data collection segment of the example. We have added a spike detector
on the "axon_terminal" locset. The :class:`arbor.single_cell_model` automatically registers all
spikes on the cell from all spike detectors on the cell and saves the times at which they occurred.

.. code-block:: python

   # Print the number of spikes.
   print(len(model.spikes), 'spikes recorded:')

   # Print the spike times.
   for s in model.spikes:
       print(s)


A more interesting result of the simulation is perhaps the output of the voltage probe previously
placed on the "custom_terminal" locset. The model saves the output of the probes as [time, value]
pairs which can then be plotted. We use `pandas` and `seaborn` for the plotting, but the user can
choose the any other library:

.. code-block:: python

   import pandas
   import seaborn

   # Plot the output of the probes
   df = pandas.DataFrame()
   for t in model.traces:
      df=df.append(pandas.DataFrame({'t/ms': t.time, 'U/mV': t.value, 'Location': str(t.location), 'Variable': t.variable}))

   seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location",col="Variable",ci=None).savefig('single_cell_detailed_result.svg')

The following plot is generated. The orange line is slightly delayed from the blue line, which is
what we'd expect because branch 4 is longer than branch 3 of the morphology. We also see 3 spikes,
corresponding to each of the current clamps placed on the cell.

.. figure:: single_cell_detailed_result.svg
    :width: 400
    :align: center

The full code
*************
You can find the full code of the example at ``python/examples/single_cell_detailed.py``.
