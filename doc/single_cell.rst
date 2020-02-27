.. _single:

Single Cell Models
==================

A *single cell model* has, as the name suggests, one cell with no network, and attached 
Arbor supports a workflow for building single cells, that can be used to
define and test individual cells that can be saved, then used in network simulations.

Building *single cell models* 

1. Defining the `morphology <single_morpho_>`_ of the cell.
2. Labeling regions and locations on the morphology.
3. Defining the mechanisms that will be applied to the cell.
4. Adding ion channels and synapses (mechanisms) to labeled regions and locations.
5. Attaching stimuli, spike detectors, event generators and probes to locations (inputs & outputs).

This guide will walk through a series of single cell models of increasing
complexity that the reader is encouraged to follow along with in Python. Links
are provide to separate documentation that covers relevant topics in more detail.

.. _single_soma:

Example 1: Single compartment cell with HH dynamics
----------------------------------------------------

.. code-block:: python

    import arbor

    # Define the morphology: a single sample with radius 3 μm.
    tree = arbor.sample_tree()
    tree.append(arbor.msample(x=0, y=0, z=0, radius=3, tag=2))

    # Define the soma and its center
    labels = arbor.label_dict({'soma': '(tag 2)', 'center': '(location 0 0.5)'})

    # Build the full cell description.
    cell = arbor.cable_cell(tree, labels)

    # Set properties of the cell.
    cell.set_properties(Vm=-40)
    cell.paint('soma', 'hh')
    cell.place('center', arbor.iclamp( 10, 2, 0.8))
    cell.place('center', arbor.spike_detector(-10))

    # Make single cell model.
    m = arbor.single_cell_model(cell)

    # Attach voltage probes, sampling at 10 kHz.
    m.probe('voltage', 'center',  10000)

    # Run simulation for 100 ms of simulated activity.
    tfinal=30
    m.run(tfinal)

Create a sample tree with a single sample of radius 3 μm
Set initial membrane potential everywhere on the cell to -40 mV.
Put hh dynamics on soma, and passive properties on the dendrites.
Attach stimuli with duration of 2 ms and current of 0.8 nA.
Add a spike detector with threshold of -10 mV.


**Everything below here is to be discarded/moved**

.. _single_morpho:

Morphology
----------

The first step in building a cell model is to define the cell's *morphology*.
Conceptually, Arbor treats morphologies as a tree of truncated frustums, with
an optional spherical segment at the root of the tree.
These are represented as a tree of sample points, where each sample has a 3D location,
a radius, and a tag, and a parent sample.

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
``sample_tree`` or ``flat_cell_builder``, or load from swc file.

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
    in Arbor the radius can vary linearly between two sample locations.

    A cylinder with equal diameter and length is used to model spherical somata
    in NEURON, which coincidently has the same surface area as a sphere of the same diameter.
    Arbor allows the user to optionally use a spherical section at the root
    of the tree to represent spherical somata.

.. note::
    In NEURON cell morphologies are constructed by creating individual sections,
    then connecting them together. In Arbor we start with an "empty"
    sample tree, to which samples are appended to build a connected morphology.

