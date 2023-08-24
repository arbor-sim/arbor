.. _cppinterconnectivity:

.. cpp:namespace:: arb

Interconnectivity
#################

.. cpp:class:: cell_connection

    Describes a connection between two cells: a pre-synaptic source and a
    post-synaptic destination. The source is typically a threshold detector on
    a cell or a spike source. The destination is a synapse on the post-synaptic cell.

    The :cpp:member:`dest` does not include the gid of a cell, this is because a
    :cpp:class:`cell_connection` is bound to the destination cell which means that the gid
    is implicitly known.

    .. cpp:member:: cell_global_label_type source

        Source end point, represented by a :cpp:type:`cell_global_label_type` which packages
        a cell gid, label of a group of sources on the cell, and source selection policy.

    .. cpp:member:: cell_local_label_type dest

        Destination end point on the cell, represented by a :cpp:type:`cell_local_label_type`
        which packages a label of a group of targets on the cell and a selection policy.
        The target cell's gid is implicitly known.

    .. cpp:member:: float weight

        The weight delivered to the target synapse.
        The weight is dimensionless, and its interpretation is
        specific to the synapse type of the target. For example,
        the `expsyn` synapse interprets it as a conductance
        with units μS (micro-Siemens).

    .. cpp:member:: float delay

        Delay of the connection (milliseconds).

.. cpp:class:: ext_cell_connection

    Describes a connection between two cells: a pre-synaptic source and a
    post-synaptic destination. The source is typically a threshold detector on
    a cell or a spike source. The destination is a synapse on the post-synaptic cell.

    The :cpp:member:`dest` does not include the gid of a cell, this is because a
    :cpp:class:`ext_cell_connection` is bound to the destination cell which means that the gid
    is implicitly known.

    .. cpp:member:: cell_remote_label_type source

        Source end point, represented by a :cpp:type:`cell_remote_label_type` which packages
        a cell gid, integral tag of a group of sources on the cell, and source selection policy.

    .. cpp:member:: cell_local_label_type dest

        Destination end point on the cell, represented by a :cpp:type:`cell_local_label_type`
        which packages a label of a group of targets on the cell and a selection policy.
        The target cell's gid is implicitly known.

    .. cpp:member:: float weight

        The weight delivered to the target synapse.
        The weight is dimensionless, and its interpretation is
        specific to the synapse type of the target. For example,
        the `expsyn` synapse interprets it as a conductance
        with units μS (micro-Siemens).

    .. cpp:member:: float delay

        Delay of the connection (milliseconds).


.. cpp:class:: gap_junction_connection

    Describes a gap junction connection between two gap junction sites. The :cpp:member:`local` site does
    not include the gid of a cell, this is because a :cpp:class:`gap_junction_connection` is bound to the local
    cell which means that the gid is implicitly known.

    .. note::

       A bidirectional gap-junction connection between two cells ``c0`` and ``c1`` requires two
       :cpp:class:`gap_junction_connection` objects to be constructed: one where ``c0`` is the
       :cpp:member:`local` site, and ``c1`` is the :cpp:member:`peer` site; and another where ``c1`` is the
       :cpp:member:`local` site, and ``c0`` is the :cpp:member:`peer` site.

    .. cpp:member:: cell_global_label_type peer

        Peer gap junction site, represented by a :cpp:type:`cell_local_label_type` which packages a cell gid,
        a label of a group of gap junction sites on the cell, and a site selection policy.

    .. cpp:member:: cell_local_label_type local

        Local gap junction site on the cell, represented by a :cpp:type:`cell_local_label_type`
        which packages a label of a group of gap junction sites on the cell and a selection policy.
        The gid of the local site's cell is implicitly known.

    .. cpp:member:: float weight

        unit-less gap junction connection weight.

.. cpp:class:: network_site_info

    A network connection site on a cell. Used for generated connections through the high-level network description.

    .. cpp:member:: cell_gid_type gid

        The cell index.

    .. cpp:member:: cell_kind kind

        The cell kind.

    .. cpp:member:: cell_tag_type label

        The associated label.

    .. cpp:member:: mlocation location

        The local location on the cell.

    .. cpp:member:: mpoint global_location

        The global location in cartesian coordinates.


.. cpp:class:: network_connection_info

    A network connection between cells. Used for generated connections through the high-level network description.

    .. cpp:member:: network_site_info src

        The source connection site.

    .. cpp:member:: network_site_info dest

        The destination connection site.


.. cpp:class:: network_value

    A network value, describing the its calculation for each connection.

   .. cpp:function:: network_value scalar(double value)

   A fixed scalar valaue.

   .. cpp:function:: network_value named(std::string name)

   A named network value in the network label dictionary.

   .. cpp:function:: network_value distance()

   The value representing the distance between source and destination.

   .. cpp:function:: network_value uniform_distribution(unsigned seed, const std::array<double, 2>& range)

   A uniform random distribution within [range_0, range_1)

   .. cpp:function:: network_value normal_distribution(unsigned seed, double mean, double std_deviation)

   A normal random distribution with given mean and standard deviation.

   .. cpp:function:: network_value truncated_normal_distribution(unsigned seed, double mean, double std_deviation, const std::array<double, 2>& range)

   A truncated normal random distribution with given mean and standard deviation. Sampled through accept-reject method to only returns values in [range_0, range_1)

   .. cpp:function:: network_value custom(custom_func_type func)

   Custom value using the provided function "func". Repeated calls with the same arguments to "func" must yield the same result.

   .. cpp:function:: network_value add(network_value left, network_value right)

   Summation of two values.

   .. cpp:function:: network_value sub(network_value left, network_value right)

   Subtraction of two values.

   .. cpp:function:: network_value mul(network_value left, network_value right)

   Multiplication of two values.

   .. cpp:function:: network_value div(network_value left, network_value right)

   Division of two values.

   .. cpp:function:: network_value min(network_value left, network_value right)

   Minimum of two values.

   .. cpp:function:: network_value max(network_value left, network_value right)

   Maximum of two values.

   .. cpp:function:: network_value exp(network_value v)

   Exponential of given value.

   .. cpp:function:: network_value log(network_value v)

   Logarithm of given value.

   .. cpp:function:: if_else(network_selection cond, network_value true_value, network_value false_value)

   if contained in selection, the true_value is used and the false_value otherwise.


.. cpp:class:: network_selection

    A network selection, describing a subset of all possible connections.

   .. cpp:function:: network_selection all()

    Select all

   .. cpp:function:: network_selection none();

    Select none

   .. cpp:function:: network_selection named(std::string name);

    Named selection in the network label dictionary

   .. cpp:function:: network_selection inter_cell();

    Only select connections between different cells

   .. cpp:function:: network_selection source_cell_kind(cell_kind kind);

    Select connections with the given source cell kind

   .. cpp:function:: network_selection destination_cell_kind(cell_kind kind);

    Select connections with the given destination cell kind

   .. cpp:function:: network_selection source_label(std::vector<cell_tag_type> labels);

    Select connections with the given source label

   .. cpp:function:: network_selection destination_label(std::vector<cell_tag_type> labels);

    Select connections with the given destination label

   .. cpp:function:: network_selection source_cell(std::vector<cell_gid_type> gids);

    Select connections with source cells matching the indices in the list

   .. cpp:function:: network_selection source_cell(gid_range range);

    Select connections with source cells matching the indices in the range

   .. cpp:function:: network_selection destination_cell(std::vector<cell_gid_type> gids);

    Select connections with destination cells matching the indices in the list

   .. cpp:function:: network_selection destination_cell(gid_range range);

    Select connections with destination cells matching the indices in the range

   .. cpp:function:: network_selection chain(std::vector<cell_gid_type> gids);

    Select connections that form a chain, such that source cell "i" is connected to the destination cell "i+1"

   .. cpp:function:: network_selection chain(gid_range range);

    Select connections that form a chain, such that source cell "i" is connected to the destination cell "i+1"

   .. cpp:function:: network_selection chain_reverse(gid_range range);

    Select connections that form a reversed chain, such that source cell "i+1" is connected to the destination cell "i"

   .. cpp:function:: network_selection intersect(network_selection left, network_selection right);

    Select connections, that are selected by both "left" and "right"

   .. cpp:function:: network_selection join(network_selection left, network_selection right);

    Select connections, that are selected by either or both "left" and "right"

   .. cpp:function:: network_selection difference(network_selection left, network_selection right);

    Select connections, that are selected by "left", unless selected by "right"

   .. cpp:function:: network_selection symmetric_difference(network_selection left, network_selection right);

    Select connections, that are selected by "left" or "right", but not both

   .. cpp:function:: network_selection complement(network_selection s);

    Invert the selection

   .. cpp:function:: network_selection random(unsigned seed, network_value p);

    Random selection using the bernoulli random distribution with probability "p" between 0.0 and 1.0

   .. cpp:function:: network_selection custom(custom_func_type func);

    Custom selection using the provided function "func". Repeated calls with the same arguments
    to "func" must yield the same result. For gap junction selection,
    "func" must be symmetric (func(a,b) = func(b,a)).

   .. cpp:function:: network_selection distance_lt(double d);

    Only select within given distance. This may enable more efficient sampling through an
    internal spatial data structure.

   .. cpp:function:: network_selection distance_gt(double d);

    Only select if distance greater then given distance. This may enable more efficient sampling
    through an internal spatial data structure.


.. cpp:class:: network_label_dict

    Dictionary storing named network values and selections.

   .. cpp:function:: network_label_dict& set(const std::string& name, network_selection s)

    Store a network selection under the given name

   .. cpp:function:: network_label_dict& set(const std::string& name, network_value v)

    Store a network value under the given name

   .. cpp:function:: std::optional<network_selection> selection(const std::string& name) const

    Returns the stored network selection of the given name if it exists. None otherwise.

   .. cpp:function:: std::optional<network_value> value(const std::string& name) const

    Returns the stored network value of the given name if it exists. None otherwise.

   .. cpp:function:: const ns_map& selections() const

    All stored network selections

   .. cpp:function:: const nv_map& selections() const

    All stored network values


.. cpp:class:: network_description

    A complete network description required for processing.

    .. cpp:member:: network_selection selection

        Selection of connections.

    .. cpp:member:: network_value weight

        Weight of generated connections.

    .. cpp:member:: network_value delay

        Delay of generated connections.

    .. cpp:member:: network_label_dict dict

        Label dictionary for named selecations and values.
