.. _tutorial_nmodl:

How to use NMODL to extend Arbor's repetoire of Ion Channels
============================================================

NMODL is Arbor's way of expressing ion channel dynamics which in turn can be
added to cable cells via the :ref:`decor` interface. This tutorial will guide
you through to create such an ion channel from scratch. Note that there are
already large selections of channels built into Arbor in addition to online
databases of ready-to-use models. Please check there first and only if you find
nothing to use or adapt, try to build your own. If you want to learn how to use
ion channels, consider other tutorials first, such as
:ref:`tutorialsinglecellrecipe` and the dynamic catalogue example.

Introducing NMODL
-----------------

NMODL is a domain specific language (DSL) for design electro-chemical dynamics
deriving from the earlier MODL language. It is used by both Arbor, Neuron, and
and CoreNeuron. Although each software has its own dialect, the core ideas are
identical. Unfortunately, documentation for both NMODL and MODL is scarce and
outdated. This tutorial aims to get you proficient in writing and adapting ion
channels for Arbor. Note that while we focus on density mechanisms here, there
is a host of behaviours customisable in Arbor through NMODL in addition to
wholly new functionality, examples include gap junctions, synapses, and voltage
clamps. NMODL can use stochastic differential equations and modify ion
concentrations.

The running example: Hodgkin-Huxley
-----------------------------------

During this tutorial, we will re-implement the classic Hodgkin-Huxley ion
channel model. You can use it directly in Arbor as ``hh`` from the default
catalogue, but for this tutorial, we pretend that it is an entirely new model.

We begin by setting up a simple model using the default ``hh`` model

.. code-block:: python

    import arbor as A
    from arbor import units as U
    import matplotlib.pyplot as plt

    # Create a single segment morphology
    tree = A.segment_tree()
    tree.append(A.mnpos, (-3, 0, 0, 3), (3, 0, 0, 3), tag=1)

    # Create (almost empty) decor
    decor = (
        A.decor()
        .paint('(all)', A.density('hh'))
        .place('(location 0 0.5)', A.iclamp(10 * U.ms, 2 * U.ms, 0.8 * U.nA), "iclamp")
    )

    # Run the model, extracting the membrane voltage
    model = A.single_cell_model(A.cable_cell(tree, decor))
    model.probe("voltage", '(location 0 0.5)', tag="Um", frequency=10 * U.kHz)
    model.run(tfinal=30 * U.ms)

    # Create a basic plot
    fg, ax = plt.subplots()
    ax.plot(m.traces[0].time, m.traces[0].value)
    ax.set_xlabel('t/ms')
    ax.set_ylabel('U/mV')
    plt.savefig('hh-01.pdf')

This should --- once run --- produce a plot like this:

.. figure:: ../images/hh-01.svg
    :width: 600
    :align: center

You can find all steps in the ``python/example/hh`` directory in Arbor's source code.

Starting out: Loading our own catalogue
---------------------------------------

Next, we have to do multiple things in parallel

1. create our own channel
2. build a catalogue containing said channel
3. update the example accordingly

We start by creating a new directory ``mod`` (the name is not important, but
will be used throughout this example) and adding a file named ``hh02.mod`` to
it. Its contents should be this:

.. code-block::

    NEURON {
        SUFFIX hh02
        NONSPECIFIC_CURRENT il
    }

    BREAKPOINT {
        il = 0
    }

We will discuss this in detail below, but for now, we will just translate and
use it in our model. Change to a shell, next to the ``mod`` directory and type:

.. code-block:: bash

   arbor-build-catalogue cat mod

and an output like this should appear (again ``cat`` is an arbitrary name we
will use from here on).

.. code-block:: bash

    Building catalogue 'cat' from mechanisms in /Users/hater/src/arbor/python/example/hh/mod
     * NMODL
       * hh02
    Catalogue has been built and copied to /Users/hater/src/arbor/python/example/hh/cat-catalogue.so

and the file ``cat-catalogue.so`` should appear in your current directory. Next, modify the Python file
like this:

.. code-block:: python

    import arbor as A
    from arbor import units as U
    import matplotlib.pyplot as plt

    # Create a single segment morphology
    tree = A.segment_tree()
    tree.append(A.mnpos, (-3, 0, 0, 3), (3, 0, 0, 3), tag=1)

    # Create (almost empty) decor
    decor = (
        A.decor()
        .paint('(all)', A.density('hh02'))
        .place('(location 0 0.5)', A.iclamp(10 * U.ms, 2 * U.ms, 0.8 * U.nA), "iclamp")
    )

    # Run the model, extracting the membrane voltage
    model = A.single_cell_model(A.cable_cell(tree, decor))

    # add our catalogue
    model.properties.catalogue = A.load_catalogue('cat-catalogue.so')

    model.probe("voltage", '(location 0 0.5)', tag="Um", frequency=10 * U.kHz)
    model.run(tfinal=30 * U.ms)

    # Create a basic plot
    fg, ax = plt.subplots()
    ax.plot(m.traces[0].time, m.traces[0].value)
    ax.set_xlabel('t/ms')
    ax.set_ylabel('U/mV')
    plt.savefig('hh-02.pdf')

This should --- once run --- produce a plot like this:

.. figure:: ../images/hh-02.svg
    :width: 600
    :align: center

You can find all steps in the ``python/example/hh`` directory in Arbor's source
code. Let's return to what just happened, it's quite a bit. First, we added our
ion channel and used ``arbor-build-catalogue`` to translate it into a form Arbor
can utilize. These collections of ion channels are --- unsurprisingly --- called
catalogues, see :ref:`mechanisms`.
