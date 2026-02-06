.. _cppadexcell:

AdEx cells
===========

.. cpp:namespace:: arb

.. cpp:class:: adex_cell

    An adaptive exponential cell.

    .. function:: adex_cell(source, target)

        Constructor: assigns the label ``source`` to the single built-in source on the cell; and assigns the
        label ``target`` to the single built-in target on the cell.

    .. cpp:member:: cell_tag_type source

        The label of the single built-in source on the cell. Used for forming
        connections from the cell in the :cpp:class:`recipe` by creating a
        :cpp:class:`connection`.

    .. cpp:member:: cell_tag_type target

        The label of the single built-in target on the cell. Used for forming
        connections to the cell in the :cpp:class:`recipe` by creating a
        :cpp:class:`connection`.

    .. cpp:member:: double V_th

        Firing threshold :math:`V_\mathrm{th} = -20\,mV` [mV].

    .. cpp:member:: double C_m

        Membrane capacitance :math:`C_\mathrm{m} = 0.28\,nF` [nF].

    .. cpp:member:: double E_L

        Resting potential :math:`E_\mathrm{L} = -70\,mV` [mV]

    .. cpp:member:: double E_R

        Reset potential :math:`E_\mathrm{R} = -70\,mV`  [mV].

    .. cpp:member:: double V_m

        Initial value of the Membrane potential :math:`V_\mathrm{m} = -70\,mV`  [mV].

    .. cpp:member:: double t_ref

        Refractory period :math:`t_\mathrm{ref} = 2.5\,ms` [ms].

    .. cpp:member:: double g

       Leak conductivity :math:`g = 0.03\,mu S` [uS].

    .. cpp:member:: double w

        :math:`w` initial value :math:`w = 0\,mA` [nA].

    .. cpp:member:: double tau

        :math:`w` decaying constant :math:`\tau = 144\,ms` [ms].

    .. cpp:member:: double a

        :math:`w` dynamics :math:`a = 0.004\,\mu S` [uS].


    .. cpp:member:: double b

       Increase in :math:`w` after emitted spike :math:`b = 0.08\,nA` [nA]
