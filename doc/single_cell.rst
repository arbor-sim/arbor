.. _single_cell:

Single Cell Models
==================

This is a practical guide for beginners on how to make and run single cell Arbor model in Python.
Building single cell models is typically the first step towards building network models.

We break down *single cell model building* into the following steps:

    1. Defining the `morphology <single_morpho_>`_ of the cell.
    2. Labeling regions and locations on the morphology.
    3. Defining the mechanisms that will be applied to the cell.
    4. Applying mechanisms to labeled regions and locations.
    5. Attaching stimuli, spike detectors, event generators, probes (inputs & outputs).
    6. Parameter sweeps.

In this guide we will provide a simple example workflow of this kind that the reader
can follow along with in Python, with links to separate guides that go into each
step in more detail for when you develop your own models.

.. Note::
    Most readers will most likely be at least familiar with NEURON. Boxes like this
    will be used to highlight differences and subtleties between NEURON and Arbor
    throughout the guide. Readers who have not used NEURON can skip these notes.

.. _single_morpho:

Morphology
----------

building the morphology there are two approaches: construct it manually using
``sample_tree`` or ``flat_cell_builder``, or load from swc file.

TODO: cover all methods here?
    - we could just ``flat_cell_builder`` because it is most comfortable for
      users coming over from NEURON.
    - have links to another page that goes into detail on all the different
      methods for morphology building. That page could take a moderately
      complicated, well-defined, morphology, and illustrate how to build
      it using all of the different methods.


