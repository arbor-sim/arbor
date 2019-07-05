.. _pycable_cell:

Python Cable Cells
====================

The interface for specifying cell morphologies with the distribution of ion channels
and synapses is a key part of the user interface. Arbor will have an advanced and user-friendly
interface for this, which is currently under construction.

To allow users to experiment will multi-compartment cells, we provide some helpers
for generating cells with random morphologies, which are documented here.

.. Warning::

    These features will be deprecated once the morphology interface has been implemented.

.. function:: make_cable_cell(seed, params)

    Construct a branching :class:`cable_cell` with a random morphology (via parameter ``seed``) and
    synapse end points locations described by parameter ``params``.

    The soma has an area of 500 μm², a bulk resistivity of 100 Ω·cm,
    and the ion channel and synapse dynamics are described by a Hodgkin-Huxley (HH) mechanism.
    The default parameters of HH mechanisms are:

    - Na-conductance        0.12 S⋅cm⁻²,
    - K-conductance         0.036 S⋅cm⁻²,
    - passive conductance   0.0003 S⋅cm⁻², and
    - passive potential     -54.3 mV

    Each cable has a diameter of 1 μm, a bulk resistivity of 100 Ω·cm,
    and the ion channel and synapse dynamics are described by a passive/ leaky integrate-and-fire model with parameters:

    - passive conductance   0.001 S⋅cm⁻², and
    - resting potential     -65 mV

    Further, a spike detector is added at the soma with threshold 10 mV,
    and a synapse is added to the mid point of the first dendrite with an exponential synapse model:

    - time decaying constant    2 ms
    - resting potential         0 mV

    Additional synapses are added based on the number of randomly generated :attr:`cell_parameters.synapses` on the cell.

    :param seed: The seed is an integral value used to seed the random number generator, for which the :attr:`arbor.cell_member.gid` of the cell is a good default.

    :param params: By default set to :class:`cell_parameters()`.

.. class:: cell_parameters

        Parameters used to generate random cell morphologies.
        Where parameters must be given as ranges, the first value is at the soma,
        and the last value is used on the last level.
        Values at levels in between are found by linear interpolation.

    .. attribute:: depth

        The maximum depth of the branch structure
        (i.e., maximum number of levels in the cell (not including the soma)).

    .. attribute:: lengths

        The length of the branch [μm], given as a range ``[l1, l2]``.

    .. attribute:: synapses

        The number of randomly generated synapses on the cell.

    .. attribute:: branch_probs

        The probability of a branch occuring, given as a range ``[p1, p2]``.

    .. attribute:: compartments

        The compartment count on a branch, given as a range ``[n1, n2]``.
