.. _pymorph:

   .. 

Cell morphology
===============

.. currentmodule:: arbor

.. data:: mnpos
    :type: int

    Value used to indicate "no parent" in :class:`segment_tree` and :class:`morphology`
    trees of segments and branches respectively.

    .. code-block:: python

        import arbor

        tree = arbor.segment_tree()

        # mnpos can be used to explicitly specify that a segment
        # is at the root of the tree. More than one segment can
        # be at the root, and they will all be joined electrically
        # at their proximal ends.
        tree.append(parent=arbor.mnpos, # attach segment to root.
                    prox=arbor.mpoint(0, 0,-5, 5),
                    dist=arbor.mpoint(0, 0, 5, 5),
                    tag=1)
        tree.append(parent=0,
                    prox=arbor.mpoint(0, 0, 5, 0.5),
                    dist=arbor.mpoint(0, 0,50, 0.2),
                    tag=3)

        # mnpos can also be used when querying a segment_tree or morphology,
        # for example the following snippet that finds all branches in the
        # morphology that are attached to the root of the morphology.
        m = arbor.morphology(tree)
        base_branches = [i for i in range(m.num_branches)
                            if m.branch_parent(i) == arbor.mnpos]

        print(base_branches)



.. class:: location

    A location on :attr:`branch`, where :attr:`pos`, in the range ``0 ≤ pos ≤ 1``,
    gives the relative position
    between the proximal and distal ends of the branch. The position is in terms
    of branch path length, so for example, on a branch of path length 100 μm ``pos=0.2``
    corresponds to 20 μm and 80 μm from the proximal and distal ends of the
    branch respectively.

    .. function:: location(branch, pos)

        Constructor.

    .. attribute:: branch
        :type: int

        The branch id of the location.

    .. attribute:: pos
        :type: float

        The relative position of the location on the branch.

.. class:: cable

    An unbranched cable that is a subset of a branch.
    The values of ``0 ≤ prox ≤ dist ≤ 1`` are the relative position
    of the cable's end points on the branch, in terms
    of branch path length. For example, on a branch of path length 100 μm, the values
    :attr:`prox` =0.2, :attr:`dist` =0.8 describe a cable that starts and
    ends 20 μm and 80 μm along the branch respectively.

    .. function:: cable(branch, prox, dist)

        Constructor.

    .. attribute:: branch
        :type: int

        The branch id of the cable.

    .. attribute:: prox
        :type: float

        The relative position of the proximal end of the cable on the branch.

    .. attribute:: dist
        :type: float

        The relative position of the distal end of the cable on the branch.

.. class:: mpoint

    A location of a cell morphology at a fixed location in space. Describes the location
    of the as three-dimensional coordinates (:attr:`x`, :attr:`y`, :attr:`z`) and
    the :attr:`radius` of the cable.

    .. attribute:: x
        :type: real

        X coordinate (μm)

    .. attribute:: y
        :type: real

        Y coordinate (μm)

    .. attribute:: z
        :type: real

        x coordinate (μm)

    .. attribute:: radius
        :type: real

        Radius of the cable (μm)

.. class:: segment

    .. attribute:: prox
        :type: mpoint

        The location and radius at the proximal end of the segment.

    .. attribute:: dist
        :type: mpoint

        The location and radius at the distal end of the segment.

    .. attribute:: tag
        :type: int

        Integer tag meta-data associated with the segment.
        Typically the tag would correspond to the SWC structure identifier:
        soma=1, axon=2, dendrite=3, apical dendrite=4, however arbitrary
        tags, including zero and negative values, can be used.

.. class:: segment_tree

    A segment tree is a description of a the segments and their connections
    Segment trees comprise a sequence of segments starting from at lease one *root* segment,
    together with a parent-child adjacency relationship where a child segment is
    distal to its parent.
    Branches in the tree occur where a segment has more than one child.
    Furthermore, a segment can not have more than one parent.
    In this manner, neuron morphologies are modeled as a *tree*, where cables that
    represent dendrites and axons can branch, but branches can not rejoin.
    A segment tree is a segment-based description of a cell's morphology.

    .. function:: segment_tree()

        Construct an empty segment tree.

    The tree is constructed by *appending* segments to the tree.
    Segments are numbered starting at 0 in the order that they are added,
    with the first segment getting id 0, the second segment id 1, and so forth.

    A segment can not be added before its parent, hence the first segment
    is always at the root. In this manner, a segment tree is
    always guaranteed to be in a correct state, with consistent parent-child
    indexing, and with *n* segments numbered from *0* to *n-1*.

    To illustrate how a segment tree is constructed by appending segments,
    take the segment tree used in the :ref:`documentation above <morph-label-seg-fig>`.


    .. figure:: ../gen-images/label_seg.svg


    Which is constructed as follows.

    .. _morph-label-seg-code:

    .. code-block:: Python

        import arbor
        from arbor import mpoint
        from arbor import mpos

        tree = arbor.segment_tree()
        # Start with a cylinder segment for the soma (with tag 1)
        tree.append(mnpos, mpoint(0,   0.0, 0, 2.0), mpoint( 4,  0.0, 0, 2.0), tag=1)
        # Construct the first section of the dendritic tree,
        # comprised of segments 1 and 2, attached to soma segment 0.
        tree.append(0,     mpoint(4,   0.0, 0, 0.8), mpoint( 8,  0.0, 0, 0.8), tag=3)
        tree.append(1,     mpoint(8,   0.0, 0, 0.8), mpoint(12, -0.5, 0, 0.8), tag=3)
        # Construct the rest of the dendritic tree.
        tree.append(2,     mpoint(12, -0.5, 0, 0.8), mpoint(20,  4.0, 0, 0.4), tag=3)
        tree.append(3,     mpoint(20,  4.0, 0, 0.4), mpoint(26,  6.0, 0, 0.2), tag=3)
        tree.append(2,     mpoint(12, -0.5, 0, 0.5), mpoint(19, -3.0, 0, 0.5), tag=3)
        tree.append(5,     mpoint(19, -3.0, 0, 0.5), mpoint(24, -7.0, 0, 0.2), tag=3)
        tree.append(5,     mpoint(19, -3.0, 0, 0.5), mpoint(23, -1.0, 0, 0.2), tag=3)
        tree.append(7,     mpoint(23, -1.0, 0, 0.2), mpoint(26, -2.0, 0, 0.2), tag=3)
        # Two segments that define the axon, with the first at the root, where its proximal
        # end will be connected with the proximal end of the soma segment.
        tree.append(mnpos, mpoint(0,   0.0, 0, 2.0), mpoint(-7,  0.0, 0, 0.4), tag=2)
        tree.append(9,     mpoint(-7,  0.0, 0, 0.4), mpoint(-10, 0.0, 0, 0.4), tag=2)

    .. method:: append(parent, prox, dist, tag)

        Append a segment to the tree.

        :return: index of the new segment
        :param int parent: index of segment
        :param mpoint prox: proximal end of the segment
        :param mpoint dist: distal end of the segment
        :param int tag: tag meta data of segment

    .. method:: append(parent, dist, tag)
        :noindex:

        Append a segment to the tree whose proximal end has the location and
        radius of the distal end of the parent segment.

        This version of append can't be used for a segment at the root of the
        tree, that is, when ``parent`` is :data:`mnpos`, in which case both proximal
        and distal ends of the segment must be specified.

        :return: index of the new segment
        :param int parent: index of segment
        :param mpoint dist: distal end of the segment
        :param int tag: tag meta data of segment

    .. method:: append(parent, x, y, z, radius, tag)
        :noindex:

        Append a segment to the tree whose proximal end has the location and
        radius of the distal end of the parent segment.

        This version of append can't be used for a segment at the root of the
        tree, that is, when ``parent`` is :data:`mnpos`, in which case both proximal
        and distal ends of the segment must be specified.

        :return: index of the new segment
        :param int parent: index of segment
        :param float x: distal x coordinate (μm)
        :param float y: distal y coordinate (μm)
        :param float z: distal z coordinate (μm)
        :param float radius: distal radius (μm)
        :param int tag: tag meta data of segment

    .. attribute:: empty
        :type: bool

        If the tree is empty (i.e. whether it has size 0)

    .. attribute:: size
        :type: int

        The number of segments.

    .. attribute:: parents
        :type: list

        A list of parent indexes of the segments.

    .. attribute:: segments
        :type: list

        A list of the segments.

.. py:class:: morphology

    A *morphology* describes the geometry of a cell as unbranched cables
    with variable radius and their associated tree structure.

    .. note::
        A morphology takes a segment tree and construct the cable branches.
        Meta data about branches and their properties that may be expensive to calculate
        is stored for fast look up during later stages of model building, and
        querying by users.

        For this reason, morphologies are read only. To change a morphology, a new
        morphology should be created using a new segment tree.

    There is one *constructor* for a morphology:

    .. function:: morphology(segment_tree)

        Construct from a segment tree.

    The morphology provides an interface for querying morphology properties:

    .. attribute:: empty
            :type: bool

            Indicates if the morphology is empty.

    .. attribute:: num_branches
            :type: int

            The number of branches in the morphology.

    .. method:: branch_parent(i)

            The parent branch of a branch.

            :param int i: branch index
            :rtype: int

    .. method:: branch_children(i)

            The child branches of a branch.

            :param int i: branch index
            :rtype: list

    .. method:: branch_segments(i)

            A list of the segments in a branch, ordered from proximal to distal.

            :param int i: branch index
            :rtype: list

.. py:function:: load_swc(filename)

    Loads the :class:`morphology` from an SWC file according to arbor's SWC specifications.
    (See the morphology concepts :ref:`page <morph-formats>` for more details).

    The samples in the SWC files are treated as the end points of segments, where a
    sample and its parent form a segment.
    The :py:attr:`tag <segment.tag>` of each segment is the
    `structure identifier <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_
    of the distal sample.
    The structure identifier of the first (root) sample is ignored, as it can only be the
    proximal end of any segment.

    .. note::
        This method does not interpret the first sample, typically associated with the soma,
        as a sphere. SWCs with single point somas are, unfortunately, reasonably common, for example
        `SONATA <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-biophysical-neuron-morphologies>`_
        model descriptions.

        Such representations are unfortunate because simulation tools like Arbor and NEURON require
        the use of cylinders or fustrums to describe morphologies, and it is not possible to
        infer how branches attached to the soma should be connected.

        The :func:`load_swc_allen` and :func:`load_swc_neuron` functions provide support for interpreting
        such SWC files.


    :param str filename: the name of the SWC file.
    :rtype: morphology

.. py:function:: load_swc_neuron(filename)

    Loads the :class:`morphology` from an SWC file according to NEURON's SWC specifications.
    Specifically:

        * The first sample must be a soma sample.
        * The soma is represented by a series of n≥1 unbranched, serially listed samples.
        * The soma is constructed as a single cylinder with diameter equal to the piecewise average diameter of all the
          segments forming the soma.
        * A single-sample soma at is constructed as a cylinder with length=diameter.
        * If a non-soma sample is to have a soma sample as its parent, it must have the most distal sample of the soma
          as the parent.
        * Every non-soma sample that has a soma sample as its parent, attaches to the created soma cylinder at its midpoint.
        * If a non-soma sample has a soma sample as its parent, no segment is created between the sample and its parent,
          instead that sample is the proximal point of a new segment, and there is a gap in the morphology (represented
          electrically as a zero-resistance wire)
        * To create a segment with a certain tag, that is to be attached to the soma, we need at least 2 samples with that
          tag.

    :param str filename: the name of the SWC file.
    :rtype: morphology


.. py:function:: load_swc_allen(filename, no_gaps=False)

    Loads the :class:`morphology` from an SWC file according to the AllenDB and Sonata's SWC specifications.
    Specifically:

        * The first sample (the root) is treated as the center of the soma.
        * The morphology is translated such that the soma is centered at (0,0,0).
        * The first sample has tag 1 (soma).
        * All other samples have tags 2, 3 or 4 (axon, apic and dend respectively)

    SONATA prescribes that there should be no gaps, however some models in AllenDB
    have gaps between the start of sections and the soma. The ``no_gaps`` flag can be
    used to enforce this requirement.

    Arbor does not support modelling the soma as a sphere, so a cylinder with length
    equal to the soma diameter is used. The cylinder is centered on the origin, and
    aligned along the z axis.
    Axons and apical dendrites are attached to the proximal end of the cylinder, and
    dendrites to the distal end, with a gap between the start of each branch and the
    end of the soma cylinder to which it is attached.

    :param str filename: the name of the SWC file.
    :param bool no_gaps: enforce that distance between soma center and branches attached to soma is the soma radius.
    :rtype: morphology

.. py:class:: place_pwlin

    A :class:`place_pwlin` object allows the querying of the 3-d location of locations and cables
    in a morphology. Refer to the C++ documentation for :cpp:type:`place_pwlin` for more details.

    .. py:function:: place_pwlin(morphology, isometry)
    .. py:function:: place_pwlin(morphology)
       :noindex:

       Construct a piecewise linear placement of the morphology in space,
       optionally applying the given isometry.

    .. py:method:: at(loc: location) -> location

       Return any single point corresponding to the :class:`location` ``loc``
       in the placement.

    .. py:method:: all_at(loc: location) -> list[location]

       Return all points corresponding to the given :class:`location` ``loc``
       the placement.

    .. py:method:: segments(cables: list[cable]) -> list[segment]

       Return any minimal collection of segments and partial segments whose
       union is coterminous with the sub-region of the morphology covered by
       the given cables in the placement.

    .. py:method:: all_segments(cables: list[cable]) -> list[segment]

       Return the maximal set of segments and partial segments whose
       union is coterminous with the sub-region of the morphology covered by
       the given cables in the placement.

.. py:class:: isometry

    Isometries represent rotations and translations in space, and can be used with
    :class:`place_pwlin` to position a morphology in an arbitrary spatial location
    and orientation. Refer to the C++ documentation for :cpp:type:`isometry` for
    more details.

    .. py::function:: isometry()

       Construct an identity isometry.

    .. py:method:: translate(x: float, y: float, z: float) -> isometry
       :staticmethod:

       Construct a translation (x, y, z) with respect to the extrinsic coordinate system.

    .. py:method:: translate(displacement: Tuple[float, float, float]) -> isometry
       :staticmethod:
       :noindex:

       Construct a translation from the elements of the given tuple.

    .. py:method:: translate(displacement: mpoint) -> isometry
       :staticmethod:
       :noindex:

       Construct a translation from the (x, y, z) components of the given :py:class:`mpoint`.

    .. py:method:: rotate(theta: float, x: float, y: float, z: float) -> isometry
       :staticmethod:

       Construct a rotation of ``theta`` radians about the axis (x, y, z) with respect to the intrinsic coordinate system.

    .. py:method:: rotate(theta: float, axiss: Tuple[float, float, float]) -> isometry
       :staticmethod:
       :noindex:

       Construct a rotation of ``theta`` radians about the axis given by the ``axis`` tuple.

    .. py:method:: __call__(point: mpoint) -> mpoint

       Apply the isometry to given point.

    .. py:method:: __call__(point: Tuple[float, float, float, ...]) -> Tuple[float, float, float, ...]
       :noindex:

       Apply the isometry to the first three components of the given tuple, interpreted as a point.

    .. py:function:: __mul__(a: isometry, b: isometry) -> isometry

       Compose the two isometries to form a new isometry that applies *b* and then applies *a*.
       Note that rotations are composed as being with respect to the *intrinsic* coordinate system,
       while translations are always taken to be with respect to the *extrinsic* absolute coordinate system.
