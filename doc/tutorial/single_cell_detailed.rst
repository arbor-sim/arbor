.. _tutorialsinglecellswc:

A detailed single-cell model
============================

We can expand on the :ref:`single segment cell example <tutorialsinglecell>` to create a more
complex single-cell model, and go through the process in more detail.

.. Note::

   **Concepts covered in this example:**

   1. Building a morphology from a :class:`arbor.segment_tree`.
   2. Building a morphology from an SWC file.
   3. Writing and visualizing region and locset expressions.
   4. Building a decor.
   5. Discretising the morphology.
   6. Setting and overriding model and cell parameters.
   7. Running a simulation and visualising the results using a :class:`arbor.single_cell_model`.

.. _tutorialsinglecellswc-cell:

The cell
********

We start by building the cell. This will be a :ref:`cable cell <cablecell>` with complex
geometry and dynamics which is constructed from 3 components:

1. A **morphology** defining the geometry of the cell.
2. A **label dictionary** storing labelled expressions which define regions and locations of
   interest on the cell.
3. A **decor** defining various properties and dynamics on these  regions and locations.
   The decor also includes hints about how the cell is to be modelled under the hood, by
   splitting it into discrete control volumes (CV).

The morphology
^^^^^^^^^^^^^^^
We begin by constructing the following morphology:

.. figure:: ../gen-images/tutorial_morph.svg
   :width: 600
   :align: center

This can be done by manually building a segment tree. The important bit here is
that ``append`` will take an id to attach to and return the newly added id. This
is exceptionally handy when building a tree structure, as we can elect to
remember or overwrite the last id. Alternatively, you could use numeric ids ---
they are just sequentially numbered by insertion order --- but we find that this
becomes tedious quickly. The image above shows the numeric ids for the specific
insertion order below, but different orders will produce the same morphology.

.. code-block:: python

    # Construct an empty segment tree.
    tree = A.segment_tree()

    # The root of the tree has no parent
    root = A.mnpos

    # The root segment: a cylindrical soma with tag 1
    # NOTE: append returns the added segment's id, which we can use to
    #       attach the next segments.
    soma = tree.append(root, (0.0, 0.0, 0.0, 2.0), (40.0, 0.0, 0.0, 2.0), tag=1)

    # Attach the first section of the dendritic tree with tag 3 to the soma
    # up to the first fork
    dend = tree.append(soma, (40.0, 0.0, 0.0, 0.8), ( 80.0,  0.0, 0.0, 0.8), tag=3)
    dend = tree.append(dend, (80.0, 0.0, 0.0, 0.8), (120.0, -5.0, 0.0, 0.8), tag=3)

    # Construct the upper part of the first fork
    # NOTE: We do not overwrite the parent here, as we need to attach the
    #       lower fork later. Instead, we use new names for this branch.
    dend_u = tree.append(dend, (120.0, -5.0, 0.0, 0.8), (200.0,  40.0, 0.0, 0.4), tag=3)
    dend_u = tree.append(dend_u, (200.0, 40.0, 0.0, 0.4), (260.0,  60.0, 0.0, 0.2), tag=3)

    # Construct the lower part of the first fork
    dend_l = tree.append(dend, (120.0, -5.0, 0.0, 0.5), (190.0, -30.0, 0.0, 0.5), tag=3)

    # Attach another fork to the last segment, ``p``.
    # Upper part
    dend_lu = tree.append(dend_l, (190.0, -30.0, 0.0, 0.5), (240.0, -70.0, 0.0, 0.2), tag=4)
    # Lower part
    dend_ll = tree.append(dend_l, (190.0, -30.0, 0.0, 0.5), (230.0, -10.0, 0.0, 0.2), tag=4)
    dend_ll = tree.append(dend_ll, (230.0, -10.0, 0.0, 0.2), (360.0, -20.0, 0.0, 0.2), tag=4)

    # Construct the axon with tag 2, attaching to the root ``mnpos``, where its
    # proximal end will be connected to the proximal end of the soma segment implicitly.
    axon = tree.append(root, (0.0, 0.0, 0.0, 2.0), (-70.0, 0.0, 0.0, 0.4), tag=2)
    axon = tree.append(axon, (-70.0, 0.0, 0.0, 0.4), (-100.0, 0.0, 0.0, 0.4), tag=2)

    # Turn the segment tree into a morphology.
    morph = arbor.morphology(tree);

The same morphology can be represented using an SWC file (interpreted according
to :ref:`Arbor's specifications <morph-formats>`). We can save the following in
``single_cell_detailed.swc``.

.. literalinclude:: ../../python/example/single_cell_detailed.swc
   :language: python

.. Note::

    SWC samples always form a segment with their parent sample. For example,
    sample 3 and sample 2 form a segment which has length = 0.
    We use these zero-length segments to represent an abrupt radius change
    in the morphology, like we see between segment 0 and segment 1 in the above
    morphology diagram.

    More information on SWC loaders can be found :ref:`here <morph-formats>`.

The morphology can then be loaded from ``single_cell_detailed.swc`` in the following way:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 11-22

We allow passing of any valid SWC file so you can easily experiment with your
own morphologies but employ some Python magic to grab the default file which
lives in the same directory as the example. In general, using a data file for
constructing morphologies should be preferred.


The label dictionary
^^^^^^^^^^^^^^^^^^^^

Next, we can define **region** and **location** expressions and give them
labels. The regions and locations are defined using an Arbor-specific DSL, and
the labels can be stored in a :class:`arbor.label_dict`.

.. Note::

   The expressions in the label dictionary don't actually refer to any concrete
   regions or locations of the morphology at this point. They are merely queries
   that can be applied to any morphology, and depending on its geometry, they
   will generate different regions and locations. However, we will show some
   figures illustrating the effect of applying these expressions to the above
   morphology in order to better visualize the final cell.

   More information on region and location expressions is available :ref:`here
   <labels>`.

The SWC file format allows the association of ``tags`` with parts of the morphology
and reserves tag values 1-4 for commonly used sections (see `here
<http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`__
for the SWC file format). In Arbor, these tags can be added to a
:class:`arbor.label_dict` using the :meth:`~arbor.label_dict.add_swc_tags`
method, which will define

.. list-table:: Default SWC Tags
   :widths: 25 25 50
   :header-rows: 1

   * - Tag
     - Label
     - Section
   * - 1
     - ``"soma"``
     - Soma
   * - 2
     - ``"axon"``
     - Axon
   * - 3
     - ``"dend"``
     - Basal dendrite
   * - 4
     - ``"apic"``
     - Apical dendrite

You can alternatively define these regions by hand, using

.. code-block:: python

  labels = arbor.label_dict({
    "soma": "(tag 1)",
    "axon": "(tag 2)",
    "dend": "(tag 3)",
    "apic": "(tag 4)",
  })

Both ways will generate the following regions when applied to the previously defined morphology:

.. figure:: ../gen-images/tutorial_tag.svg
  :width: 800
  :align: center

  From left to right: regions "soma", "axon", "dend" and "last"

We can also define a region that represents the whole cell; and to make things a
bit more interesting, a region that includes the parts of the morphology that
have a radius greater than 1.5 μm. This is done in the following way:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 30-31

This will generate the following regions when applied to the previously defined morphology:

.. figure:: ../gen-images/tutorial_all_gt.svg
  :width: 400
  :align: center

  Left: region "all"; right: region "gt_1.5"

By comparing to the original morphology, we can see region "gt_1.5" includes all of segment 0 and part of
segment 9.

Finally, let's define a region that includes two already defined regions: "last" and "gt_1.5". This can
be done as follows:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 32-33

This will generate the following region when applied to the previously defined morphology:

.. figure:: ../gen-images/tutorial_custom.svg
  :width: 200
  :align: center

Our label dictionary so far only contains regions. We can also add some **locations**. Let's start
with a location that is the root of the morphology, and the set of locations that represent all the
terminal points of the morphology.

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 35-37

This will generate the following **locsets** (sets of one or more locations) when applied to the
previously defined morphology:

.. figure:: ../gen-images/tutorial_root_term.svg
  :width: 400
  :align: center

  Left: locset "root"; right: locset "terminal"

To make things more interesting, let's select only the terminal points which belong to the
previously defined "custom" region and, separately, the terminal points which belong to the
"axon" region:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 38-41

This will generate the following 2 locsets when applied to the previously defined morphology:

.. figure:: ../gen-images/tutorial_custom_axon_term.svg
  :width: 400
  :align: center

  Left: locset "custom_terminal"; right: locset "axon_terminal"

.. note::

   We show the use of the label dictionary here, but everywhere a label is
   valid, you can use the DSL expression directly, so

   .. code-block:: python

       decor.paint('(all)', arbor.density("pas"))

   is perfectly acceptable, as is

   .. code-block:: python

       all = '(all)'
       decor.paint(all, arbor.density("pas"))


The decorations
^^^^^^^^^^^^^^^

With the key regions and location expressions identified and labelled, we can start to
define certain features, properties, and dynamics of the cell. This is done through a
:class:`arbor.decor` object, which stores a mapping of these "decorations" to certain
region or location expressions.

.. Note::

  Similar to the label dictionary, the decor object is merely a description of how an abstract
  cell should behave, which can then be applied to any morphology, and have a different effect
  depending on the geometry and region/locset expressions.

  More information on decors can be found :ref:`here <cablecell-decoration>`.

The decor object can have default values for properties, which can then be
overridden on specific regions. It is in general better to explicitly set all
the default properties of your cell, to avoid the confusion of having
simulator-specific default values. This will, therefore, be our first step:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 48-62

We have set the default initial membrane voltage mV, the default initial
temperature, the default axial resistivity, and the default membrane
capacitance. Also, the initial properties of the *na* and *k* ions have been
changed. They will be utilized by the density mechanisms that we will be adding
shortly. For both ions we set the default initial concentration and external
concentration measures; and we set the default initial reversal potential. For
the *na* ion, we additionally indicate that the progression on the reversal
potential during the simulation will be dictated by the `Nernst equation
<https://en.wikipedia.org/wiki/Nernst_equation>`_. In the case that defaults are
not set at the cell level, there is also a global default, which we will define later.

It happens, however, that we want the temperature of the "custom" region defined
in the label dictionary earlier to be colder, and the initial voltage of the
"soma" region to be higher. We can override the default properties by *painting*
new values on the relevant regions using :meth:`arbor.decor.paint`.

.. literalinclude:: ../../python/example/single_cell_detailed.py
  :language: python
  :lines: 63-65

With the default and initial values taken care of, we now add some density
mechanisms. Let's *paint* a *pas* density mechanism everywhere on the cell using
the previously defined "all" region; an *hh* density mechanism on the "custom"
region, and an *Ih* density mechanism on the "end" region. The *Ih* mechanism
has a custom 'gbar' parameter.

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 66-69

The decor object is also used to *place* stimuli and threshold detectors on the
cell using :meth:`arbor.decor.place`. We place 3 current clamps of 2 nA on the
"root" locset defined earlier, starting at time = 10, 30, 50 ms and lasting 1ms
each. As well as threshold detectors on the "axon_terminal" locset for voltages
above -10 mV. Every placement gets a label. The labels of detectors and synapses
are used to form connections from and to them in the recipe.

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 70-74

.. Note::

   The number of individual locations in the ``'axon_terminal'`` locset depends
   on the underlying morphology and the number of axon branches in the
   morphology. The number of detectors that get added to the cell is equal to
   the number of locations in the locset, and the label ``'detector'`` refers to
   all of them collectively. If we want to refer to a single detector from the
   group (to form a network connection for example), we need a
   :py:class:`arbor.selection_policy`.

Finally, there's one last property that impacts the behavior of a model: the
discretisation. Cells in Arbor are simulated as discrete components called
control volumes (CV). The size of a CV has an impact on the accuracy of the
results of the simulation. Usually, smaller CVs are more accurate because they
simulate the continuous nature of a neuron more closely.

The user controls the discretisation using a :class:`arbor.cv_policy`. There are
a few different policies to choose from, and they can be composed with one
another. In this example, we would like the "soma" region to be a single CV, and
the rest of the morphology to be comprised of CVs with a maximum length of 1 μm:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 75-76

Finally, we create the cell.

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 79-80

The model
*********

Having created the cell, we construct an :class:`arbor.single_cell_model`.

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 82-83

The global properties
^^^^^^^^^^^^^^^^^^^^^

The global properties of a single-cell model include:

1. The **mechanism catalogue**: A mechanism catalogue is a collection of density
   and point mechanisms. Arbor has 3 built-in mechanism catalogues: ``default``,
   ``allen`` and ``bbp``. The mechanism catalogue in the global properties of
   the model must include the catalogues of all the mechanisms painted on the
   cell decor. The default is to use the ``default_catalogue``.

2. The default **parameters**: The initial membrane voltage; the initial
   temperature; the axial resistivity; the membrane capacitance; the ion
   parameters; and the discretisation policy.

.. Note::

   You may notice that the same parameters can be set both at the cell level and
   at the model level. This is intentional. The model parameters apply to all
   the cells in a model, whereas the cell parameters apply only to that specific
   cell.

   The idea is that the user is able to define a set of global properties for
   all cells in a model which can then be overridden for individual cells, and
   overridden yet again on certain regions of the cells.

   You may now be wondering why this is needed for the `single cell model` where
   there is only one cell by design. You can use this feature to ease moving
   from simulating a set of single-cell models to simulating a network of these
   cells. For example, a user may choose to individually test several single
   cell models before simulating their interactions. By using the same global
   properties for each *model*, and customizing the *cell* global properties, it
   becomes possible to use the cell descriptions of each cell, unchanged, in a
   larger network model.

Earlier in the example we mentioned that it is better to explicitly set all the
default properties of your cell, while that is true, it is better yet to set the
default properties of the entire model:

.. _tutorialsinglecellswc-gprop:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 85-94

We set the same properties as we did earlier when we were creating the *decor*
of the cell, except for the initial membrane voltage, which is -65 mV as opposed
to -55 mV.

During the decoration step, we also made use of 3 mechanisms: *pas*, *hh* and
*Ih*. As it happens, the *pas* and *hh* mechanisms are in the default Arbor
catalogue, whereas the *Ih* mechanism is in the "allen" catalogue. We can extend
the default catalogue as follows:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 96-100

Now all three mechanisms in the *decor* object have been made available to the model.

The probes
^^^^^^^^^^

The model itself is ready for simulation, except that the only output we would
be able to measure at this point is the spikes from the threshold detectors
placed in the decor.

The :class:`arbor.single_cell_model` can also measure the voltage on specific
locations of the cell. We can indicate the location we would like to probe using
labels from the :class:`label_dict`:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 102-104

The simulation
^^^^^^^^^^^^^^

The cell and model descriptions are now complete and we can run the simulation:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 106-107

The results
^^^^^^^^^^^

Finally we move on to the data collection segment of the example. We have added
a threshold detector on the "axon_terminal" locset. The
:class:`arbor.single_cell_model` automatically registers all spikes on the cell
from all threshold detectors on the cell and saves the times at which they
occurred.

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 109-112

A more interesting result of the simulation is perhaps the output of the voltage
probe previously placed on the "custom_terminal" locset. The model saves the
output of the probes as [time, value] pairs which can then be plotted. We use
`pandas` and `seaborn` for the plotting, but the user can choose any other
library:

.. literalinclude:: ../../python/example/single_cell_detailed.py
   :language: python
   :lines: 114-

The following plot is generated. The orange line is slightly delayed from the
blue line, which is what we'd expect because branch 4 is longer than branch 3 of
the morphology. We also see 3 spikes, corresponding to each of the current
clamps placed on the cell.

.. figure:: single_cell_detailed_result.svg
    :width: 400
    :align: center

The full code
*************
You can find the full code of the example at ``python/examples/single_cell_detailed.py``.
