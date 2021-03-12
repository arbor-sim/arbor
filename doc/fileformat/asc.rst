.. _formatasc:

Neurlucida ASCII format
~~~~~~~~~~~~~~~~~~~~~~~

Arbor has support for reading cable cell morphologies described using the
`Neurlucida ASCII file format <https://www.mbfbioscience.com/help/pdf/NL9.pdf>`_
or ``.asc`` file format.

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

    As a result, 
    Please open an issue if:

      * you have an ``.asc`` file that Arbor can't parse;
      * there is meta data, such as spine locations, that is missing in the output.

    and we will 
    add support to our parser for the features in your file. Over time Arbor's support.

Soma / CellBody
""""""""""""""""

The soma, or CellBody, is described in one of three different methods (that we are aware of) in
an ASCII file.

  1. As a CellBody statement containing a single location and radius, which models **a sphere**.
  2. As a CellBody statement containing an ubranched sequence of locations that define **a single contour**.
  3. As multiple CellBody statements, each defining a contour, that describe the soma as **a stack of contours**.

Arbor supports description methods 1 and 2, and support for method 3 can be added on request
(open an issue).

In each case, the soma is modeled as a cylinder with diameter equal to it's length, centred
at the centre of the soma, and oriented along the z axis.

For a **spherical** soma, the centre and diameter are that of the sphere. For
a **contour**, the centre is the centroid of the locations that define the contour,
and the radius is the average distance of the centre to the locations on the countour.

API
"""

* :ref:`Python <pyasc>`
* :ref:`C++ <cppasc>`

