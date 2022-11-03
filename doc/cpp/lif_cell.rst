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

    .. cpp:member:: double tau_m

        Membrane potential decaying constant [ms].

    .. cpp:member:: double V_th

        Firing threshold [mV].

    .. cpp:member:: double C_m

        Membrane capacitance [pF].

    .. cpp:member:: double E_L

        Resting potential [mV].

    .. cpp:member:: double E_R

        Reset potential [mV].

    .. cpp:member:: double V_m

        Initial value of the Membrane potential [mV].

    .. cpp:member:: double t_ref

        Refractory period [ms].
