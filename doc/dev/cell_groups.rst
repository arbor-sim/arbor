.. _cell_groups:

Cell groups
===========

Cell groups represent a union of cells of a single *kind* simulated in lockstep.
In a sense, their existence is an optimisation, since parts of the internal
state and computations can be shared between cell in single group. The currently
most complicated cell group is the one for cable cells, called ``mc_cell_group``
(``mc`` stands for multi-compartment, used in older parts of Arbor), so we will
focus on this type here.

Cell groups are created by domain decomposition methods on consideration of soft
(like performance optimisation) and hard (cells connected by gap junctions must
be in the same group) constraints.

Cable Cell group ``mc_cell_group``
----------------------------------

Cable cell groups have backing store in ``shared_state`` (given the
introduction, we now understand that the ``shared`` stands for 'shared' between
cell in a group). In this data set, we also collect the private data of
mechanisms. One thing to watch out for here is that instances of the same
mechanism on a cell group will be collated.

Let us examine an example of such a cell group; assume the following

1. the group comprises two cells with a total of 9 CVs
   1. cell 0 has 4 CVs
   2. cell 1 has 5 CVs
2. ``pas`` has been painted on three regions and collated
   1. region 0 covers CVs [0 1 2]
   2. region 1 covers CV 5
   3. region 2 covers CV 7
3. ``pas`` has two parameters ``g`` (conductance density) and ``E`` (resting potential)

.. code::

  - shared_state
    - mechanisms
      - id 0
        - name   "pas"
        - width  5
        - parameters
          - id 0
            - name "g"
            - values [* * * * *]
          - id 1
            - name "E"
            - values [* * * * *]
        - index  [0 1 2 5 7]
                  / | | |  \
                 / / /  |  |
                / / /    \  \
    - voltage [* * * * * * * * *]

Mechanisms access their view of the cell group data via the
``arb_mechanism_ppack`` structure. To continue with our example, the ``pas``
mechanism would iterate through its view on the cell group voltage to
compute the current density ``i`` like this

.. code::

   for ix in 0..width
     # Obtain parameters
     g  = ppack.parameters["g"][ix]
     E  = ppack.parameters["E"][ix]
     # Fetch voltage, note the indirection
     cv = ppack.index[ix]
     u  = ppack.voltage[cv]
     # Write outgoing current
     ppack.i[ix] = g*(u - E)

In general, cell group wide quantities (like ``voltage`` here) need to be
indexed via ``ppack.index``. Note, that the layout of parameters in ``ppack`` is
this in reality:

.. code::

   - ppack
     - parameters   [g g g g g E E E E E]

When using NMODL, we translate names like ``g`` to offsets into the parameter array
at compile time. Handwritten mechanisms need to do this manually.
