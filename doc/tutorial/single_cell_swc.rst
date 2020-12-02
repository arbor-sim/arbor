.. _tutorialsinglecellswc:

A detailed branchy cell
---------------------

We can build on the :ref:`single segment cell example <gs_single_cell>` to create a more
complex single cell model, in terms of the morphology, dynamics, and discretisation control.

Let's begin with constructing the following morphology:

.. figure:: ../gen-images/example_morph0.svg
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
    from arbor import mpoint
    from arbor import mpos

    #(1) Define the morphology from an SWC file

    morph = arbor.load_swc_arbor("morph.swc")