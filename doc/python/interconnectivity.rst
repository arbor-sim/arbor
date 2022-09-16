.. _pyinterconnectivity:

Interconnectivity
#################

.. currentmodule:: arbor

.. class:: connection

    Describes a connection between two cells, defined by source and destination end points (that is pre-synaptic and
    post-synaptic respectively), a connection weight and a delay time.

    The :attr:`dest` does not include the gid of a cell, this is because a :class:`arbor.connection` is bound to the
    destination cell which means that the gid is implicitly known.

    .. function:: connection(source, destination, weight, delay)

        Construct a connection between the :attr:`source` and the :attr:`dest` with a :attr:`weight` and :attr:`delay`.

    .. attribute:: source

        The source end point of the connection (type: :class:`arbor.cell_global_label`, which can be initialized with a
        (gid, label) or a (gid, (label, policy)) tuple. If the policy is not indicated, the default
        :attr:`arbor.selection_policy.univalent` is used).

    .. attribute:: dest

        The destination end point of the connection (type: :class:`arbor.cell_local_label` representing the label of the
        destination on the cell, which can be initialized with just a label, in which case the default
        :attr:`arbor.selection_policy.univalent` is used, or a (label, policy) tuple). The gid of the cell is
        implicitly known.

    .. attribute:: weight

        The weight delivered to the target synapse. It is up to the target mechanism to interpret this quantity.
        For Arbor-supplied point processes, such as the ``expsyn`` synapse, a weight of ``1`` corresponds to an
        increase in conductivity in the target mechanism of ``1`` μS (micro-Siemens).

    .. attribute:: delay

        The delay time of the connection [ms]. Must be positive.

    .. note::

        An minimal full example of a connection reads as follows:
        (see :ref:`network tutorial <tutorialnetworkring>` for a more comprehensive example):

        .. code-block:: python

            import arbor

            # Create two locset labels, describing the endpoints of the connection.
            labels = arbor.label_dict()
            labels['synapse_site'] = '(location 1 0.5)'
            labels['root'] = '(root)'

            # Place 'expsyn' mechanism on "synapse_site", and a threshold detector at "root"
            decor = arbor.decor()
            decor.place('"synapse_site"', 'expsyn', 'syn')
            decor.place('"root"', arbor.threshold_detector(-10), 'detector')

            # Implement the connections_on() function on a recipe as follows:
            def connections_on(gid):
               # construct a connection from the "detector" source label on cell 2
               # to the "syn" target label on cell gid with weight 0.01 and delay of 10 ms.
               src  = (2, "detector") # gid and locset label of the source
               dest = "syn" # gid of the destination is determined by the argument to `connections_on`.
               w    = 0.01  # weight of the connection. Correspondes to 0.01 μS on expsyn mechanisms
               d    = 10 # delay in ms
               return [arbor.connection(src, dest, w, d)]

.. class:: gap_junction_connection

    Describes a gap junction between two gap junction sites.

    The :attr:`local` site does not include the gid of a cell, this is because a :class:`arbor.gap_junction_connection`
    is bound to the destination cell which means that the gid is implicitly known.

    .. note::

       A bidirectional gap-junction between two cells ``c0`` and ``c1`` requires two
       :class:`gap_junction_connection` objects to be constructed: one where ``c0`` is the
       :attr:`local` site, and ``c1`` is the :attr:`peer` site; and another where ``c1`` is the
       :attr:`local` site, and ``c0`` is the :attr:`peer` site.

    .. function::gap_junction_connection(peer, local, weight)

        Construct a gap junction connection between :attr:`peer` and :attr:`local` with weight :attr:`weight`.

    .. attribute:: peer

        The gap junction site: the remote half of the gap junction connection (type: :class:`arbor.cell_global_label`,
        which can be initialized with a (gid, label) or a (gid, label, policy) tuple. If the policy is not indicated,
        the default :attr:`arbor.selection_policy.univalent` is used).

    .. attribute:: local

        The gap junction site: the local half of the gap junction connection (type: :class:`arbor.cell_local_label`
        representing the label of the destination on the cell, which can be initialized with just a label, in which case
        the default :attr:`arbor.selection_policy.univalent` is used, or a (label, policy) tuple). The gid of the
        cell is implicitly known.

    .. attribute:: weight

        The unit-less weight of the gap junction connection.

.. class:: threshold_detector

    A threshold detector, generates a spike when voltage crosses a threshold. Can be used as source endpoint for an
    :class:`arbor.connection`.

    .. attribute:: threshold

        Voltage threshold of threshold detector [mV]
