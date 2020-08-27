.. _single:

Single Cell Models
==================

Building and testing detailed models of individual cells, then optimizing their parameters
is usually the first step in building models with multi-compartment cells.
Arbor supports a *single cell model* workflow for this purpose, which is a good way to
introduce Arbor's cell modeling concepts and approach.

This guide will walk through a series of single cell models of increasing
complexity that the reader is encouraged to follow along with in Python. Links
are provide to separate documentation that covers relevant topics in more detail.

.. _single_soma:

Example 1: Single compartment cell with HH dynamics
----------------------------------------------------

The most trivial representation of a cell in Arbor is to model the entire cell as a sphere.
The following model shows the steps required to construct a model of a spherical cell with
radius 3 μm, Hodgkin–Huxley dynamics and a current clamp stimulus, then run the model for
30 ms.

The first step is to construct the cell. In Arbor, the abstract representation used to define
a cell with branching "cable" morphology is a ``cable_cell``, which holds a description
of the cell's morphology, named regions and locations on the morphology, and descriptions of
ion channels, synapses, spike detectors and electrical properties.

Our "single-compartment HH cell" has a trivial morphology and dynamics, so the steps to
create the ``cable_cell`` that represents it are quite straightforward:

.. code-block:: python

    import arbor

    # (1) Create a morphology with a single segment of length=diameter=6 μm
    tree = arbor.segment_tree()
    tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

    # (2) Define the soma and its center
    labels = arbor.label_dict({'soma':   '(tag 1)',
                               'center': '(location 0 0.5)'})

    # (3) Create cell and set properties
    cell = arbor.cable_cell(tree, labels)
    cell.set_properties(Vm=-40)
    cell.paint('soma', 'hh')
    cell.place('center', arbor.iclamp( 10, 2, 0.8))
    cell.place('center', arbor.spike_detector(-10))

Arbor's cell morphologies are constructed from a :ref:`segment tree<morph-segment_tree>`,
which is a list of segments, which are tapered cones with a *tag*.
Step **(1)** above shows how the spherical cell is represented using a single segment.

Cell builders need to refer to *regions* and *locations* on a cell morphology.
Arbor uses a domains specific language (DSL) to describe regions and locations,
which are given labels. In step **(2)** a dictionary of labels is created
with two labels:

* ``soma`` defines a *region* with ``(tag  2)``. Note that this corresponds to the ``tag`` parameter that was used to define the single segment in step (1).
* ``center`` defines a *location* at ``(location 0 0.5)``, which is the mid point ``0.5`` of branch ``0``, which corresponds to the center of the soma on the morphology defined in Step (1).

In step **(3)** the cable cell is constructed by combining the segment tree with
the named regions and locations.

* Set initial membrane potential everywhere on the cell to -40 mV.
* Use HH dynamics on soma.
* Attach stimuli with duration of 2 ms and current of 0.8 nA.
* Add a spike detector with threshold of -10 mV.

Arbor can simulate networks with multiple individual cells, connected together in a network.
Single cell models do not require the full *recipe* interface used to describing such
network models, with many unique cells, network and gap junctions.
Arbor provides a ``single_cell_model`` helper that wraps a cell description, and provides
an interface for recording 


.. code-block:: python

    # (4) Make single cell model.
    m = arbor.single_cell_model(cell)

    # (5) Attach voltage probe sampling at 10 kHz (every 0.1 ms).
    m.probe('voltage', 'center', frequency=10000)

    # (6) Run simulation for 100 ms of simulated activity.
    m.run(tfinal=100)

Step (4) instantiates the single cell model using our single-compartment cell.
To record variables from the model three pieces of information are prov
* 


**Everything below here is to be discarded/moved**

.. _single_morpho:

Morphology
----------

The first step in building a cell model is to define the cell's *morphology*.
Conceptually, Arbor treats morphologies as a tree of truncated frustums, with
an optional spherical segment at the root of the tree.
These are represented as a tree of segments, where each segment is defined
by two end points with radius, and a tag, and a parent segment to which it is attached.

Let's start with a simple "ball and stick" model cell.

.. container:: example-code

    .. code-block:: python

        import arbor
        builder = arbor.flat_cell_builder()

        # Start with a spherical segment with radius 10 μm.
        # Label this segment 'soma'.
        p = builder.add_sphere(radius=10, name='soma')

        # Attach a cable to the soma with length 100 μm and constant radius 4 μm.
        q = builder.add_cable(parent=p, length=100, radius=4, name='dend')

        # Attach two dendrites to the first of length 50 μm, that taper from 4 μm to 2 μm.
        p = builder.add_cable(parent=q, length=50, radius=(4,2), name='dend')
        p = builder.add_cable(parent=q, length=50, radius=(4,2), name='dend')


Building the morphology there are two approaches: construct it manually using
``segment_tree`` or ``flat_cell_builder``, or load from swc file.

TODO: cover all methods here?
    - we could just ``flat_cell_builder`` because it is most comfortable for
      users coming over from NEURON.
    - have links to another page that goes into detail on all the different
      methods for morphology building. That page could take a moderately
      complicated, well-defined, morphology, and illustrate how to build
      it using all of the different methods.

NEURON erratum
------------------------------

These should probably be combined into a single section that describes the differences
between Arbor and NEURON, because the alternative is to keep stopping the
narative to point out the difference with NEURON, instead of explaining what
Arbor is from a fresh start.

.. note::
    Most readers will be familiar with NEURON. Boxes like this
    will be used to highlight differences between NEURON and Arbor
    throughout the guide.

    NEURON users will recognise that Arbor uses many similar concepts, and
    an effort has been made to use the same nomenclature wherever possible.

    Arbor takes a more structured approach to model building,
    from morphology descriptions up to network connectivity, to allow model
    descriptions that are more scalable and portable.

.. note::
    NEURON represents morphologies as a tree of cylindrical *segments*, whereas
    in Arbor the radius can vary linearly along segments.

    A cylinder with equal diameter and length is used to model spherical somata
    in NEURON, which coincidently has the same surface area as a sphere of the same diameter.
    Arbor allows the user to optionally use a spherical section at the root
    of the tree to represent spherical somata.

.. note::
    In NEURON cell morphologies are constructed by creating individual sections,
    then connecting them together. In Arbor we start with an "empty"
    segment tree, to which segments are appended to build a connected morphology.

1. Defining the `morphology <single_morpho_>`_ of the cell.
2. Labeling regions and locations on the morphology.
3. Defining the mechanisms that will be applied to the cell.
4. Adding ion channels and synapses (mechanisms) to labeled regions and locations.
5. Attaching stimuli, spike detectors, event generators and probes to locations (inputs & outputs).

