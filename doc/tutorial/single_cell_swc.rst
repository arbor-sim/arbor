.. _tutorialsinglecellswc:

A detailed branchy cell
-----------------------

We can build on the :ref:`single segment cell example <gs_single_cell>` to create a more
complex single cell model, in terms of the morphology, dynamics, and discretisation control.

We start by building the cell, a process which can be split into 3 parts:

1. Define the geometry of the cell using a **morphology**.
1. Define the regions and locations of interest on the cell using a **label dictionary**
2. Set various properties and dynamics on these labeled regions and locations, using
   a **decor**. This includes hints about how the cell is to be modeled under the hood, by
   splitting it into discrete control volumes (CV).

The morpholohgy
^^^^^^^^^^^^^^^
Let's begin with constructing the following morphology:

.. figure:: ../gen-images/example0_morph.svg
   :width: 400
   :align: center

This can be done by manually building a segment tree:

.. code-block:: python

    import arbor
    from arbor import mpoint
    from arbor import mnpos

    #(1) Define the morphology by manually building a segment tree

    tree = arbor.segment_tree()

    # Start with segment 0: a cylindrical soma with tag 1
    tree.append(mnpos, mpoint(0.0, 0.0, 0.0, 2.0), mpoint( 4.0, 0.0, 0.0, 2.0), tag=1)
    # Construct the first section of the dendritic tree with tag 3,
    # comprised of segments 1 and 2, attached to soma segment 0.
    tree.append(0,     mpoint(4.0, 0.0, 0.0, 0.8), mpoint( 8.0,  0.0, 0.0, 0.8), tag=3)
    tree.append(1,     mpoint(8.0, 0.0, 0.0, 0.8), mpoint(12.0, -0.5, 0.0, 0.8), tag=3)
    # Construct the rest of the dendritic tree: segments 3, 4 and 5.
    tree.append(2,     mpoint(12.0, -0.5, 0.0, 0.8), mpoint(20.0,  4.0, 0.0, 0.4), tag=3)
    tree.append(3,     mpoint(20.0,  4.0, 0.0, 0.4), mpoint(26.0,  6.0, 0.0, 0.2), tag=3)
    tree.append(2,     mpoint(12.0, -0.5, 0.0, 0.5), mpoint(19.0, -3.0, 0.0, 0.5), tag=3)
    # Construct a special region of the tree made of segments 6, 7, and 8
    # differentiated from the rest of the tree using tag 4.
    tree.append(5,     mpoint(19.0, -3.0, 0.0, 0.5), mpoint(24.0, -7.0, 0.0, 0.2), tag=4)
    tree.append(5,     mpoint(19.0, -3.0, 0.0, 0.5), mpoint(23.0, -1.0, 0.0, 0.2), tag=4)
    tree.append(7,     mpoint(23.0, -1.0, 0.0, 0.2), mpoint(26.0, -2.0, 0.0, 0.2), tag=4)
    # Construct segments 9 and 10 that make up the axon with tag 2.
    # Segment 9 is at the root, where its proximal end will be connected to the
    # proximal end of the soma segment.
    tree.append(mnpos, mpoint( 0.0, 0.0, 0.0, 2.0), mpoint( -7.0, 0.0, 0.0, 0.4), tag=2)
    tree.append(9,     mpoint(-7.0, 0.0, 0.0, 0.4), mpoint(-10.0, 0.0, 0.0, 0.4), tag=2)

    morph = arbor.morphology(tree);

The same morphology can be represented using an SWC file (interpreted according
to :ref:`Arbor's specifications <morph-formats>`). We can save the following in
``morph.swc``.

.. code-block:: python

    # id,  tag,      x,      y,      z,      r,    parent
        1     1     0.0     0.0     0.0     2.0        -1  # seg0 prox / seg9 prox
        2     1     4.0     0.0     0.0     2.0         1  # seg0 dist
        3     3     4.0     0.0     0.0     0.8         2  # seg1 prox
        4     3     8.0     0.0     0.0     0.8         3  # seg1 dist / seg2 prox
        5     3    12.0    -0.5     0.0     0.8         4  # seg2 dist / seg3 prox
        6     3    20.0     4.0     0.0     0.4         5  # seg3 dist / seg4 prox
        7     3    26.0     6.0     0.0     0.2         6  # seg4 dist
        8     3    12.0    -0.5     0.0     0.5         5  # seg5 prox
        9     3    19.0    -3.0     0.0     0.5         8  # seg5 dist / seg6 prox / seg7 prox
       10     4    24.0    -7.0     0.0     0.2         9  # seg6 dist
       11     4    23.0    -1.0     0.0     0.2         9  # seg7 dist / seg8 prox
       12     4    26.0    -2.0     0.0     0.2        11  # seg8 dist
       13     2    -7.0     0.0     0.0     0.4         1  # seg9 dist / seg10 prox
       14     2   -10.0     0.0     0.0     0.4        13  # seg10 dist

.. note::
    SWC samples always form a segment with their parent segment. For example,
    sample 3 and sample 2 form a segment which has length = 0.
    We use these zero-length segments to represent an abrupt radius change
    in the morphology, like we see between segment 0 and segment 1 in the above
    morphology.

The morphology can then be loaded from ``morph.swc`` in the following way:

.. code-block:: python

    import arbor

    #(1) Read the morphology from an SWC file

    morph = arbor.load_swc_arbor("morph.swc")

The label dictionary
^^^^^^^^^^^^^^^^^^^^

Next, we can define **region** and **location** expressions and give them labels.
The regions and locations are defined using an Arbor-specific DSL, and the labels
can be stored in a **label dictionary**.

.. Note::

   The expressions in the label dictionary don't actually refer to any concrete regions
   or locations of the morphology at this point. They are merely descriptions that can be
   applied to any morphology, and depending on its geometry, they will generate different
   regions and locations. However, we will show some figures illustrating the effect of
   applying these expressions to the above morphology, in order to better visualize the
   final model.

First, we can define some **regions**, These are continuous parts of the morphology,
They can correspond to full segments or parts of segments. Our morphology already has some
pre-established regions determined by the ``tag`` parameter of the segments. They are
defined as follows:

.. code-block:: python

    import arbor

    #(1) Read the morphology from an SWC file

    morph = arbor.load_swc_arbor("morph.swc")

    #(2) Create a label dictionary

    labels = arbor.label_dict()

    # Add labels for tag 1, 2, 3, 4
    labels['soma'] = '(tag 1)'
    labels['axon'] = '(tag 2)'
    labels['dend'] = '(tag 3)'
    labels['last'] = '(tag 4)'

This will generate the following regions when applied to the previously defined morphology:

.. figure:: ../gen-images/example0_tag.svg
  :width: 800
  :align: center

  From left to right: regions "soma", "axon", "dend" and "last"

We can also define a region that represents the whole cell; and to make things a bit more interesting,
a region that includes parts of the morphology that have a radius greater than 1.5 μm. This is done in
the following way:

.. code-block:: python

    # Add a label for a region that includes the whole cell
    labels['all'] = '(all)'

    # Add a label for the parts of the cell with radius greater than 1.5 μm.
    labels['gt_1.5'] = '(radius-ge (region "all") 1.5)'

This will generate the following regions when applied to the previously defined morphology:

.. figure:: ../gen-images/example0_all_gt.svg
  :width: 400
  :align: center

  Left: region "all"; right: region "gt_1.5"

Looking at the morphology definition, we can see region "gt_1.5" includes all of segment 0 and part of
segment 9.

Finally, let's define a final region that includes our two custom regions: "last" and "gt_1.5". This can
be done as follows:

.. code-block:: python

    # Join regions "last" and "gt_1.5"
    labels['custom'] = '(join (region "last") (region "gt_1.5"))'

This will generate the following region when applied to the previously defined morphology:

.. figure:: ../gen-images/example0_tag4_gt.svg
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

.. figure:: ../gen-images/example0_root_term.svg
  :width: 400
  :align: center

To make things more interesting, let's only select the points of the morphology that are the
terminal points, but which also belong to the previously defined region "custom":

.. code-block:: python

    # Add a labels for the terminal locations in the "custom" region:
    labels['custom_terminals'] = '(restrict (locset "terminal") (region "custom"))'

This will generate the following locset when applied to the previously defined morphology:

.. figure:: ../gen-images/example0_tag4_term.svg
  :width: 200
  :align: center


The Decorations
^^^^^^^^^^^^^^^

With the key regions and location expressions identified and labeled, we can start to
define certain features, properties and dynamics on the cell. This is done through a
*decor* object, which stores a mapping of these "decorations" to certain region or location
expressions.

.. Note::

  Similar to the label dictionary, the decor object is merely a description of how an abstract
  cell should behave, which can then be applied to any morphology, and have a different effect
  depending on the geometry.

The decor object can have default values for properties, which can then be overridden for on specific
regions. It is in general better to explicitly set all the default properties of your cell,
to avoid the confusion to having simulator-specific default values. This will therefore be our first
step:

.. code-block:: python

    import arbor

    #(1) Read the morphology from an SWC file

    morph = arbor.load_swc_arbor("morph.swc")

    #(2) Create and populate the label dictionary

    labels = arbor.label_dict()
    labels['soma'] = '(tag 1)'
    # ...

    #(3) Create a decor object
    decor = arbor.decor()

    # Set the default properties
    decor.set_property(Vm =-55, tempK=300, rL=35.4, cm=0.01)
    decor.set_ion('na', int_con=10,   ex_con=140, rev_pot=50, method='nernst/na')
    decor.set_ion('k',  int_con=54.4, ex_con=2.5, rev_pot=-77)

With that, we have set the default initial membrane voltage to -55 mV; the default initial
temperature to 300 K; the default axial resistivity to 35.4 Ω·cm; and the default membrane
capacitance to 0.01 F/m².

We also set the initial properties of the *na* and *k* ions because they will be utilized
by the mechanisms we define in the coming sections.

For both ions we set the default initial concentration and external concentration measures in mM;
and we set the default initial reversal potential in mV. For the *na* ion, we additionally indicate
the the progression on the reversal potential during the simulation will be dictated by the *nernst*
equation.

It happens, however, that we want the temperature of the "custom" region defined in the label
dictionary earlier to be colder, and the initial voltage of the "soma" region to be higher.
We can override the default properties by *painting* new values on the relevant regions:

.. code-block:: python

    # Override default parameters on certain regions

   decor.paint('"custom"', tempK=270)
   decor.paint('"soma"', Vm=-50)

With our default and initial values taken care of, we can add some density mechanisms. Let's place
a *pas* mechanism everywhere on the cell using the previously defined "all" region; an *hh* mechanism
on the "custom" region; and an *Ih* mechanism on the "dend" region. The *Ih* mechanism is explicitly
constructed in order to change the default values of its 'gbar' parameter.


.. code-block:: python

   # Paint density mechanisms on certain regions

   from arbor import mechanism as mech

   decor.paint('"all"', 'pas')
   decor.paint('"custom"', 'hh')
   decor.paint('"dend"',  mech('Ih', params={'gbar', 0.001}))

The decor object is also used to place stimuli and spike detectors on the cell. We *place* 3 current
clamps of 0.5 nA on the "root" locset defined earlier, starting at time = 3, 5, 7 ms and lasting 1ms each.
As well as spike detectors on the "custom_terminals" locset for voltages above -10 mV:

 .. code-block:: python

   # Place stimuli and spike detectors on certain locsets

   decor.place('"root"', arbor.iclamp(3, 1, current=0.5))
   decor.place('"root"', arbor.iclamp(5, 1, current=0.5))
   decor.place('"root"', arbor.iclamp(7, 1, current=0.5))
   decor.place('"custom_terminals"', arbor.spike_detector(-10))

Finally, there's one last property that impacts the behavior of a model: the discretisation.
Cells in Arbor are simulated as discrete components called control volumes (CV). The size of
a CV has an impact on the accuracy of the results of the simulation. Usually, smaller CVs
are more accurate because they simulate the continuous nature of a neuron more accurately.

The user controls the discretisation using a *cv_policy*. There are a few different policies to choose
from. and they can be composed with one another. In this example, we would like the "soma" region
to be a single CV, and the rest of the morphology to be comprised of CVs with a maximum length of 1 μm:

.. code-block:: python

   # Single CV for the "soma" region
   soma_policy = arbor.cv_policy_single('"soma"')

   # CVs with max length = 1 μm as default
   dflt_policy = arbor.cv_policy_max_extent(1.0)

   # default policy everywhere except the soma
   policy = dflt_policy | soma_policy

   decor.discretization(policy)

Constructing the cell
^^^^^^^^^^^^^^^^^^^^^

With the 3 main components defined, we can now create the cell.

Here is the code so far:

.. code-block:: python

   import arbor
   from arbor import mechanism as mech

   #(1) Read the morphology from an SWC file.

   morph = arbor.load_swc_arbor("morph.swc")

   #(2) Create and populate the label dictionary.

   labels = arbor.label_dict()

   # Regions:

   labels['soma'] = '(tag 1)'
   labels['axon'] = '(tag 2)'
   labels['dend'] = '(tag 3)'
   labels['last'] = '(tag 4)'

   labels['all'] = '(all)'

   labels['gt_1.5'] = '(radius-ge (region "all") 1.5)'
   labels['custom'] = '(join (region "last") (region "gt_1.5"))'

   # Locsets:

   labels['root']     = '(root)'
   labels['terminal'] = '(terminal)'
   labels['custom_terminals'] = '(restrict (locset "terminal") (region "custom"))'

   # (3) Create and populate the decor.

   decor = arbor.decor()

   # Set the default properties.

   decor.set_property(Vm =-55, tempK=300, rL=35.4, cm=0.01)
   decor.set_ion('na', int_con=10,   ex_con=140, rev_pot=50, method='nernst/na')
   decor.set_ion('k',  int_con=54.4, ex_con=2.5, rev_pot=-77)

   # Override the defaults.

   decor.paint('"custom"', tempK=270)
   decor.paint('"soma"',   Vm=-50)

   # Paint density mechanisms.

   decor.paint('"all"', 'pas')
   decor.paint('"custom"', 'hh')
   decor.paint('"dend"',  mech('Ih', params={'gbar', 0.001}))

   # Place stimuli and spike detectors.

   decor.place('"root"', arbor.iclamp(3, 1, current=0.5))
   decor.place('"root"', arbor.iclamp(5, 1, current=0.5))
   decor.place('"root"', arbor.iclamp(7, 1, current=0.5))
   decor.place('"custom_terminals"', arbor.spike_detector(-10))

   # Create the cell.

   cell = arbor.cable_cell(morph, labels, decor)