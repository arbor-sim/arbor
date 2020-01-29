.. _single_cell:

Single Cell Models
==================

Building single cell models is the first step towards building network models.
Arbor breaks down *single cell model building* into the following steps:

1. Defining the `morphology <single_morpho_>`_ of the cell.
2. Labeling regions and locations on the morphology.
3. Defining the mechanisms that will be applied to the cell.
4. Applying mechanisms to labeled regions and locations.
5. Attaching stimuli, spike detectors, event generators, probes (inputs & outputs).

In this guide we will provide a simple example workflow of this kind that the reader
can follow along with in Python, with links to separate guides that go into each
step in more detail for when you develop your own models.

.. note::
    Most readers will most likely be familiar with NEURON. Boxes like this
    will be used to highlight differences between NEURON and Arbor
    throughout the guide. Readers who have not used NEURON can skip these notes.

    NEURON users will recognise that Arbor uses many of the same concepts, and
    a concerted effort has been made to use the same nomenclature wherever possible.

    Arbor takes a more structured approach to model building,
    from morphology descriptions up to network connectivity, to allow model
    descriptions that are more scalable and portable.

.. _single_morpho:

Morphology
----------

The first step in building a cell model is to define the cell's *morphology*.
Conceptually, Arbor describes morphologies as a tree of connect truncated frustrums, with an optional spherical segment at the root of the tree.
Internally Arbor represents morphologies as a tree of sample points, where
each sample has a 3D location, a radius and a tag.

.. note::
    NEURON represents morphologies as a tree of cylindrical *sections*, whereas
    in Arbor the radius can vary linearly along a section.

    A cylinder with equal diameter and length is used to model spherical somata
    in NEURON, which has the same surface area as a sphere of the same diameter.
    Arbor allows the user to optionally use a spherical section at the root
    of the tree to represent spherical somata.

.. note::
    In NEURON cell morphologies are constructed by creating individual sections,
    then connecting them together. Arbor's approach starts with an "empty"
    sample tree, to which samples are appended to build a connected morphology.

Let's start with a simple "ball and stick" model cell.

.. container:: example-code

    .. code-block:: python

        import arbor
        arbor.flat_cell_builder()

        # Add a spherical soma with radius 10 μm.
        p = builder.add_sphere(radius=10, name='soma')

        # Attach a cable to the soma with lenght 100 μm and constant raidus 4 μm.
        q = builder.add_cable(parent=p, length=100, radius=4, name="dend")

        # Attach two dendrites to the first of length 50 μm, that taper from 4 μm to 2 μm.
        p = builder.add_cable(parent=q, length=50, radius=(4,2), name="dend")
        p = builder.add_cable(parent=q, length=50, radius=(4,2), name="dend")



building the morphology there are two approaches: construct it manually using
``sample_tree`` or ``flat_cell_builder``, or load from swc file.

TODO: cover all methods here?
    - we could just ``flat_cell_builder`` because it is most comfortable for
      users coming over from NEURON.
    - have links to another page that goes into detail on all the different
      methods for morphology building. That page could take a moderately
      complicated, well-defined, morphology, and illustrate how to build
      it using all of the different methods.

Labeling Regions and Locations
------------------------------


