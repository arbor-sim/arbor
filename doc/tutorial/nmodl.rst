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

Store it in ``step-01.py`` and --- once run --- is should produce a plot like
this:

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
catalogues, see :ref:`mechanisms`. We pulled this into our model by loading and
assigning to the model.

Next, let's look at the output graph. We observe a sudden jump in potential
during the period the current clamp is active. As Arbor's model for a single CV
cable cell is :math:`\partial_t U_m = i_e - i_m` (for multi-CV cells we have
additional terms that can be neglected here, see :ref:`cable_cell`), this
behaviour is expected. The current clamp provides a positive :math:`i_e` and our
ion channel model is supplying the transmembrane current :math:`i_m = 0`. To
understand the latter part, consider the channel model file we just added

.. code-block::

    NEURON {
        SUFFIX hh02
        NONSPECIFIC_CURRENT il
    }

This is the ``NEURON`` block declaring the channel's name, here ``hh02``, which
is used when adding channels from a catalogue. Files that put ``SUFFIX`` in
front of the name are converted to density channels, as opposed to synapses
(``POINT_PROCESS``) and gap junctions (``JUNCTION_PROCESS``). In addition to
naming the channel, we also need to set up all variables used to interface with
Arbor, namely ion currrents, ion concentrations, ion reversal potentials, and
non-ion currents. While the ion variables follow a rigid naming scheme, which we
will discuss later, non-ion currents can be freely named after
``NONSPECIFIC_CURRENT``. We chose ``il`` here, alluding to 'leak current'.
Semantically, these currents are considered to be unassociated to any specific
ion and thus can represent all ion currents we do not model explicitly as a lump
sum. When computing ``i_m`` for the cable equation above, Arbor takes the sum
over all non-specific and ion currents across all ion channels on the
current CV. We will revisit the ``NEURON`` multiple times later on, but for now
we turn to

.. code-block::

    BREAKPOINT {
        il = 0
    }

During the integration of the cable equation, Arbor will evaluate this block to
update its internal picture of the currrents, i.e. to calculate ``i_m``. This
occurs at an unspecified moment of the execution and might even be done multiple
times, so we need to take care not to depend on execution order. We are
_expected_, yet not forced by the tooling to update all such outputs, so, again,
some care is needed.

Stepping Stone: Leak
--------------------

As you might have anticipated, our next step is to produce a finite current to
counteract any disturbance in the membrane potential. So, we start by adding a
new mechanism to ``mod``, called ``hh03``, which is just a copy of ``hh02.mod``.
Next, adjust ``SUFFIX hh02`` to ``SUFFIX hh03``. Similarly copy ``step-02.py``
to and change

.. code-block:: python

    decor = (
        A.decor()
        .paint('(all)', A.density('hh03'))
        .place('(location 0 0.5)', A.iclamp(10 * U.ms, 2 * U.ms, 0.8 * U.nA), "iclamp")
    )

as well as ``plt.savefig(hh-03.pdf)``. From on out, we'll assume the following
steps are completed at the beginning of each new section:

1. Copy ``step-n.py`` to ``step-(n+1).py``

   - update all references to ``hhn`` to ``hh(n+1)``
   - update the output image to ``hh-(n+1).pdf``
2. Copy ``mod/hhn.mod`` to ``mod/hh(n+1).mod``

   - change the name to ``SUFFIX hh(n+1)``
3. Start editing the new NMODL and Python files.

   - After each change to the NMODL file, you'll need to call ``arbor-build-catalogue cat mod``

Keep this in mind, while we start altering the NMODL file to produce a more
sensible current. Let's start with the current itself

.. code-block::

    BREAKPOINT {
        il = gl*(v - el)
    }

this will pull the membrane potential ``v`` towards a resting potential ``el``
since our reduced cable equation is now :math:`\partial_t U_m = i_e - g_l*(U_m -
E_l)`. The membrane potential is available in NMODL as a read-only built-in
symbol ``v`` and can be used in any ion channel. However, we need a way to set
the resting potential ``el`` and the conductivity ``gl``. This is accomplished
by adding a new block to the NMODL file:

.. code-block::

    PARAMETER {
        gl =   0.0003 (S/cm2)
        el = -54.3    (mV)
    }

these parameters have an optional default value and a likewise optional unit.
Both are helpful to have, though. The units chosen internally by Arbor come
together such that the conductivity _must_ have units ``S/cm2``. Note that there
is neither a check nor a conversion of units, the annotation serves purely as a
reminder to us.

We have now recreated the leak current from the HH neuron model, which is one of
three currents needed. Before we turn to the other two, though, we'll apply some
polish. Variables declared in ``PARAMETER`` blocks can be set in the call to
``paint``, like so:

.. code-block:: python

    decor = (
        A.decor()
        .paint('(all)', A.density('hh03', g=0.0005, el=-70))
        .place('(location 0 0.5)', A.iclamp(10 * U.ms, 2 * U.ms, 0.8 * U.nA), "iclamp")
    )

To enable this, we need to tell NMODL, that each CV will have its own value of
``gl`` and ``el``, via

.. code-block::

    NEURON {
        SUFFIX hh02
        NONSPECIFIC_CURRENT il
        RANGE gl, el
    }

Without this addition, there would be one, global copy for each, which could be
set by writing

.. code-block:: python

    decor = (
        A.decor()
        .paint('(all)', A.density('hh03/el=-70,gl=0.0005'))
        .place('(location 0 0.5)', A.iclamp(10 * U.ms, 2 * U.ms, 0.8 * U.nA), "iclamp")
    )

instead. Parameters are either ``GLOBAL`` or ``RANGE``, never both. The
difference is subtle and non-existant for our single CV. The rule of thumb is
that if you expect that a parameter varies smoothly across the neuron, make it
``RANGE`` and if you expect discrete, clearly delineated regions with
dicontinuous values, go for ``GLOBAL``. If in doubt, choose ``RANGE``.
Performance-wise, ``GLOBAL`` is more efficient as ``RANGE`` parameter consume
one memory location per CV _and_ require one memory access each. ``GLOBAL``
requires one location and access _regardless_ of CV count. So, if speed is an
issue, consider ``GLOBAL`` unless required otherwise.
