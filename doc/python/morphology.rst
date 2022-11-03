.. _pymorph:

Cable cell morphology
=====================

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

    .. method:: split_at(id)

        Split a segment_tree ``T`` into a pair of subtrees ``(L, R)`` such that
        ``R`` is the subtree of ``T`` that starts at the given id and L is ``T``
        without ``R``. Splitting above the root ``mnpos`` returns ``(T, {})``.

    .. method:: join_at(id, other)

        Join two subtrees ``L`` and ``R`` at a given ``id`` in ``L``, such that
        ``join_at`` is inverse to ``split_at`` for a proper choice of ``id``.
        The join point ``id`` must be in ``L``.

    .. method:: tag_roots(tag)

        Get IDs of roots of region with a particular ``tag`` in the segment tree, i.e.
        segments whose parent is either :data:`mnpos` or a segment with a different
        tag.

    .. method:: apply_isometry(iso)

        Apply an :py:class:`isometry` to the segment tree, returns the transformed tree as a copy.
        Isometries are rotations around an arbritary axis and/or translations; they can
        be instantiated using ``translate`` and ``rotate`` and combined
        using the ``*`` operator.

        :return: new tree
        :param iso: isometry

    .. method:: equivalent(other)

        Two trees are equivalent if

        1. the root segments' ``prox`` and ``dist`` points and their ``tags``
           are identical.
        2. recursively: all sub-trees starting at the current segment are
           equivalent.

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

    .. py:method:: closest(x: real, y: real, z: real) -> tuple[mpoint, real]

        Find the closest location to p. Returns the location and its distance from the input coordinates.

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

.. _pymorph-cv-policies:

Discretisation and CV policies
------------------------------

The set of boundary points used by the simulator is determined by a
:ref:`CV policy <morph-cv-policies>`. These are objects of type
:cpp:class:`cv_policy`, which has the following public methods:

.. py:class:: cv_policy

   .. attribute:: domain

       A read only string expression describing the subset of a cell morphology
       (region) on which this policy has been declared.

   CV policies can be :ref:`composed <morph-cv-composition>` with
   ``+`` and ``|`` operators.

   .. code-block:: Python

       # The plus operator applies
       policy = arbor.cv_policy_single('"soma"') + cv_policy('"dend"')

       # The | operator uses CVs of length 10 μm everywhere, except
       # on the soma, to which a single CV policy is applied.
       policy = arbor.cv_policy_max_extent(10) | cv_policy_single('"soma"')

Specific CV policy objects are created by functions described below.
These all take a ``region`` parameter that restrict the
domain of applicability of that policy; this facility is useful for specifying
differing discretisations on different parts of a cell morphology. When a CV
policy is constrained in this manner, the boundary of the domain will always
constitute part of the CV boundary point set.

.. py:function:: cv_policy_single(domain='(all)')

    Use one CV for the whole cell, or one for each connected component of the
    supplied domain.

    .. code-block:: Python

        # Use one CV for the entire cell (a single compartment model)
        single_comp = arbor.cv_policy_single()

        # Use a single CV for the soma.
        single_comp_soma = arbor.cv_policy_single('"soma"')

    :param str domain: The region on which the policy is applied.

.. py:function:: cv_policy_explicit(locset, domain='(all)')

    Use the provided locset as control volume boundaries.

    .. code-block:: Python

        # Place CV boundaries midway every branch.
        midbranch_cvp = arbor.cv_policy_explicit('(on-branches 0.5)')

        # Place CV boundaries at 10 random positions on the soma.
        random_soma_cvp = arbor.cv_policy_explicit('(uniform (tag 3) 0 9 0)','"soma"')

    :param str locset: The locset on which CV boundaries are placed.
    :param str domain: The region on which the policy is applied.

.. py:function:: cv_policy_every_segment(domain='(all)')

    Use every sample point in the morphology definition as a CV boundary, optionally
    restricted to the supplied domain. Each fork point in the domain is
    represented by a trivial CV.

    :param str domain: The region on which the policy is applied.

.. py:function:: cv_policy_fixed_per_branch(cv_per_branch, domain='(all)')

    For each branch in each connected component of the domain (or the whole cell,
    if no domain is given), evenly distribute boundary points along the branch so
    as to produce exactly ``cv_per_branch`` CVs.

    :param int cv_per_branch: The number of CVs per branch.
    :param str domain: The region on which the policy is applied.

.. py:function:: cv_policy_max_extent(max_extent, domain='(all)')

    As for :py:func:`cv_policy_fixed_per_branch`, save that the number of CVs on any
    given branch will be chosen to be the smallest number that ensures no
    CV will have an extent on the branch longer than ``max_extent`` micrometres.

    :param float max_etent: The maximum length for generated CVs.
    :param str domain: The region on which the policy is applied.

CV discretization as mcables
----------------------------

It is often useful for the user to have a detailed view of the CVs generated for a
given morphology and :ref:`cv-policy <pymorph-cv-policies>`. For example, while
debugging and fine-tuning model implementations, it can be helpful to know how many CVs
a cable-cell is comprised of, or how many CVs lie on a certain region of the cell.

The following classes and functions allow the user to inspect the CVs of a cell or
region.

.. py:class:: cell_cv_data

   Stores the discretisation data of a cable-cell in terms of CVs and the :py:class:`cables <cable>`
   comprising each of these CVs.

   .. py:method:: cables(idx) -> list[cable]

      Returns a list of :py:class:`cable` representing the CV at a given index ``idx``.

   .. py:method:: children(idx) -> list[int]

      Returns a list of the indices of the CVs representing the children of the CV at index ``idx``.

   .. py:method:: parent(idx) -> int

      Returns the index of the CV representing the parent of the CV at index ``idx``.

   .. py:attribute:: int num_cv

      Returns the total number of CVs on the cell.

.. py:function:: cv_data(cell) -> optional<cell_cv_data>

   Constructs a :py:class:`cell_cv_data` object representing the CVs comprising the cable-cell according
   to the :py:class:`cv_policy` provided in the :py:class:`decor` of the cell. Returns ``None`` if no
   :py:class:`cv_policy` was provided in the decor.

   :param cable_cell cell: The cable-cell.
   :rtype: optional<:py:class:`cell_cv_data`>

.. py:function:: intersect_region(reg, cv_data, integrate_along) -> list[idx, proportion]

   Returns a list of tuples ``[idx, proportion]`` identifying the indices (``idx``) of the CVs from the
   ``cv_data`` argument that lie in the provided region ``reg``, and how much of each CV belongs to that
   region (``proportion``). The ``proportion`` is either the area proportion or the length proportion,
   chosen according to the ``integrate_along`` argument.

   :param str reg: The region on the cable-cell represented as s-expression or a label from the
       label-dictionary of the cell.
   :param cell_cv_data cv_data: The cv_data of a cell.
   :param string integrate_along: Either "area" or "length". Decides whether the proportion of a
       CV is measured according to the area or length of the CV.
   :rtype: list[idx, proportion]

.. _pyswc:

SWC
---

.. py:function:: load_swc_arbor(filename, raw=False)

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
        as a sphere. SWC files with single point somas are common, for example
        `SONATA <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-biophysical-neuron-morphologies>`_
        model descriptions.

        Such representations are challenging to consistently interpret in different
        simulation tools because they require heuristics and, often undocumented, rules
        for how to interpret the connectin of axons and dendrites to the soma.

        The :func:`load_swc_neuron` function provides support for loading
        SWC files according to the interpretation used by NEURON.


    :param str filename: the name of the SWC file.
    :param bool raw: return segment_tree instead of morphology?
    :rtype: morphology or segment_tree

.. py:function:: load_swc_neuron(filename, raw=False)

    Loads the :class:`morphology` from an SWC file according to NEURON's ``Import3D``
    interpretation of the SWC specification.
    See :ref:`the SWC file documention <formatswc-neuron>` for more details.

    :param str filename: the name of the SWC file.
    :param bool raw: return segment_tree instead of morphology?
    :rtype: morphology or segment_tree

.. _pyneuroml:

NeuroML
-------

.. py:class:: neuroml_morph_data

    A :class:`neuroml_morphology_data` object contains a representation of a morphology defined in
    NeuroML.

    .. py:attribute:: cell_id
       :type: optional<str>

       The id attribute of the cell that was used to find the morphology in the NeuroML document, if any.

    .. py:attribute:: id
       :type: str

       The id attribute of the morphology.

    .. py:attribute:: group_segments
       :type: dict[str, list[long]]

       A map from each segment group id to its corresponding collection of segments.

    .. py:attribute:: morphology
       :type: morphology

       The morphology associated with the :class:`neuroml_morph_data` object.

    .. py:method:: segments

       Returns a label dictionary with a region entry for each segment, keyed by the segment id (as a string).

       :rtype: label_dict

    .. py:method:: named_segments

       Returns a label dictionary with a region entry for each name attribute given to one or more segments.
       The region corresponds to the union of all segments sharing the same name attribute.

       :rtype: label_dict

    .. py:method:: groups

       Returns a label dictionary  with a region entry for each defined segment group.

       :rtype: label_dict

.. py:class:: neuroml

    A :class:`neuroml` object represent NeuroML documents, and provides methods for the identification and
    translation of morphology data.

    An implementation limitation restricts valid segment id values to those which can be represented by an
    unsigned long long value.

    The ``allow_spherical_root`` optional parameter below, if set to true, will instruct the parser to
    interpret a zero-length constant radius root segment as denoting a spherical segment, and this will
    in turn be represented in the resultant morphology by a cylinder of equivalent surface area.

   .. py:method:: neuroml(filename)

      Build a NeuroML document representation from the supplied file contents.

      :param str filename: the name of the NeuroML file.

   .. py:method:: cell_ids()

      Return the id of each ``<cell>`` element defined in the NeuroML document.

      :rtype: list[str]

   .. py:method:: morphology_ids()

      Return the id of each top-level ``<morphology>`` element defined in the NeuroML document.

      :rtype: list[str]

   .. py:method:: morphology(morph_id, allow_spherical_root=false)

      Returns a representation of the top-level morphology with the supplied morph_id if it could be found.
      Parse errors or an inconsistent representation will raise an exception.

      :param str morph_id: ID of the top-level morphology.
      :param bool allow_spherical_root: Treat zero length root segments especially.
      :rtype: optional(neuroml_morph_data)

   .. py:method:: cell_morphology(cell_id, allow_spherical_root=false)

      Returns a representation of the morphology associated with the cell with the supplied cell_id if it
      could be found. Parse errors or an inconsistent representation will raise an exception.

      :param str morph_id: ID of the cell.
      :param bool allow_spherical_root: Treat zero length root segments especially.
      :rtype: optional(neuroml_morph_data)

.. _pyasc:

Neurolucida
-----------

.. py:class:: asc_morphology

   The morphology and label dictionary meta-data loaded from a Neurolucida ASCII ``.asc`` file.

   .. py:attribute:: morphology

       The cable cell morphology.

   .. py:attribute:: segment_tree

       The raw segment tree.

   .. py:attribute:: labels

       The labeled regions and locations extracted from the meta data. The four canonical regions are labeled
       ``'soma'``, ``'axon'``, ``'dend'`` and ``'apic'``.

.. py:function:: load_asc(filename)

   Loads the :class:`asc_morphology` from a :ref:`Neurolucida ASCII file <formatasc>`.

   .. code-block:: Python

       import arbor

       # Load morphology and labels from file.
       asc = arbor.load_asc('granule.asc')

       # Construct a cable cell.
       cell = arbor.cable_cell(asc.morphology, arbor.decor(), asc.labels)


   :param str filename: the name of the input file.
   :rtype: asc_morphology
