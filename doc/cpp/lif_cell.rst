.. _cpplifcell:

LIF cells
===========

.. cpp:namespace:: arb

.. cpp:class:: lif_cell

    A benchmarking cell (leaky integrate-and-fire), used by Arbor developers to test communication performance,
    with neuronal parameters:

    .. cpp:function:: lif_cell(cell_tag_type source, cell_tag_type target)

        Constructor: assigns the label ``source`` to the single built-in source on the cell; and assigns the
        label ``target`` to the single built-in target on the cell.

    .. cpp:member:: cell_tag_type source

        The label of the single built-in source on the cell. Used for forming connections from the cell in the
        :cpp:class:`recipe` by creating a :cpp:class:`connection`.

    .. cpp:member:: cell_tag_type target

        The label of the single built-in target on the cell. Used for forming connections to the cell in the
        :cpp:class:`recipe` by creating a :cpp:class:`connection`.

    .. cpp:member:: const arb::units::quantity& tau_m

        Membrane potential decaying constant [ms]. Must be finite and positive.

    .. cpp:member:: const arb::units::quantity& V_th

        Firing threshold [mV], must be finite.

    .. cpp:member:: const arb::units::quantity& C_m

        Membrane capacitance [pF], must be finite and positive.

    .. cpp:member:: const arb::units::quantity& E_L

        Resting potential [mV], must be finite.

    .. cpp:member:: const arb::units::quantity& E_R

        Reset potential [mV], must be finite.

    .. cpp:member:: const arb::units::quantity& V_m

        Initial value of the Membrane potential [mV], must be finite.

    .. cpp:member:: const arb::units::quantity& t_ref

        Refractory period [ms]. Must be finite and positive.
