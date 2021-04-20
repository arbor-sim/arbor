.. _formatasc:

Neurolucida ASCII
~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Name", "File extension", "Read", "Write"

   "Neurolucida", "``asc``", "✓", "✗"

Arbor has support for reading cable cell morphologies described using the
`Neurolucida ASCII file format <https://www.mbfbioscience.com/help/pdf/NL9.pdf>`_.

Because ASCII files contain both a morphology description and meta data, the
loader returns both a morphology, and a label dictionary that describes regions
and locsets from the meta data.

.. warning::
    The ASCII file format has no formal specification, and describes not only the cell
    morphology and features like spines, but rich meta data about markers and colors.
    Not all of the information is relevant for simulation.

    Arbor's ASCII importer discards descriptions that it determines are
    not relevant for simulation (e.g. color meta-data attached to a dendrite).
    However Arbor will throw an error when it encounters content that it can't interpret,
    unlike some other readers that will ignore anything they don't recognise.

    Because Arbor does not yet recognise all common asc file patterns, it is possible that your
    model might return an error message when you try to load it in Arbor.
    We add support for features as they are bought to our attention, because we must rely on users
    in place of a formal specification.

    Please open an `issue <https://github.com/arbor-sim/arbor/issues>`_ if:

      * you have an ``.asc`` file that Arbor can't parse;
      * or there is meta data, such as spine locations, that you would like to see in the output;

    and we will add support for your ASCII files.

Soma / CellBody
""""""""""""""""

The soma, or CellBody, is described in one of three different methods in an ASCII file:

  1. As a CellBody statement containing a single location and radius, which models **a sphere**.
  2. As a CellBody statement containing an unbranched sequence of locations that define **a single contour**.
  3. As multiple CellBody statements, each defining a contour, that describe the soma as **a stack of contours**.

Arbor supports description methods 1 and 2 following the `neuromporpho policies <http://neuromorpho.org/SomaFormat.html>`_.
Currently multiple contours in method 3 are not supported, and if you need support make
the request with an `issue <https://github.com/arbor-sim/arbor/issues>`_.

In each case, the soma is modeled as a cylinder with diameter equal to its length, centred
at the centre of the soma, and oriented along the y axis.

For a **spherical** soma, the centre and diameter are that of the sphere. For
a **contour**, the centre is the centroid of the locations that define the contour,
and the radius is the average distance of the centre to the locations on the countour.

API
"""

* :ref:`Python <pyasc>`
* :ref:`C++ <cppasc>`

