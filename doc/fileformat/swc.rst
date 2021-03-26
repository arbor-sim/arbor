.. _formatswc:

SWC
~~~

Arbor supports reading morphologies described using the
`SWC <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_ file format.

SWC files may contain comments, which are stored as metadata. And a blank line anywhere in the file is
interpreted as end of data.

The description of the morphology is encoded as a list of samples with an id,
an `x,y,z` location in space, a radius, a tag and a parent id. Arbor parses these samples, performs some checks,
then generates a morphology according to one of three possible interpretations.

The SWC file format specifications are not very detailed, which has lead different simulators to interpret
SWC files in different ways, especially when it comes to the soma. Arbor has its own an interpretation that
is powerful and simple to understand at the same time. However, we have also developed functions that will
interpret SWC files similarly to how the NEURON simulator would, and how the Allen Institute would.

Despite the differences between the interpretations, there is a common set of checks that are always performed
to validate an SWC file:

* Check that there are no duplicate ids.
* Check that the parent id of a sample is less than the id of the sample.
* Check that the parent id of a sample refers to an existing sample.

In addition, all interpretations agree that a *segment* is (in the common case) constructed between a sample and
its parent and inherits the tag of the sample; and if more than 1 sample have the same parent, the parent sample
is interpreted as a fork point in the morphology, and acts as the proximal point to a new branch for each of its
"child" samples. There a couple of exceptions to these rules which are listed below.

Arbor interpretation
""""""""""""""""""""
In addition to the previously listed checks, the arbor interpretation explicitly disallows SWC files where the soma is
described by a single sample. It constructs the soma from 2 or more samples, forming 1 or more segments. A *segment* is
always constructed between a sample and its parent. This means that there are no gaps in the resulting morphology.

Arbor has no magic rules or transformations for the soma. It can be a single branch or multiple branches; segments
of a different tag can connect to its distal end, proximal end or anywhere in the middle. For example, to create a
morphology with a single segment soma; a single segment axon connected to one end of the soma; and a single segment
dendrite connected to the other end of the soma, the following swc file can be used:


.. literalinclude :: example.swc
   :language: python
   :linenos:

Samples 1 and 2 will form the soma; samples 1 and 3 will form the axon, connected to the soma at the proximal end;
samples 2 and 4 will form the dendrite, connected to the soma at the distal end. The morphology will look something
like this:

.. figure:: ../gen-images/swc_morph.svg
   :width: 400
   :align: center

NEURON interpretation
"""""""""""""""""""""
The NEURON interpretation was obtained by experimenting with the ``Import3d_SWC_read`` function.

.. note:

   There were bugs in the Import 3D method for SWC files that were addressed with the
   release of NEURON 8. Arbor uses the new interpretation used by NEURON 8.

NEURON interprets the SWC file using the interpretation used by the
`Neuromorpho site <http://neuromorpho.org/SomaFormat.html>`_, with the exception
with how a single-sample soma is interpretted:
* The soma is constructed from a cylinder extended along the x axis, not the y axis.
* The soma is constructed from three points, the first at ``x=x0-r``, the second with
  ``x=x0`` and the third at ``x=x0+r``, to form a single section, with all dendrites, axons
  and apical dendrites attached to the center of the soma with "zero resistance wires".
  * If the Neuromorpho interpretation was followed exactly the soma would be two
    cylinders that have their proximal ends at the soma center, extended along the
    y axis, to form two NEURON sections.

Allen interpretation
""""""""""""""""""""
In addition to the previously mentioned checks, the Allen interpretation expects a single-sample soma to be the first
sample of the file and to be interpreted as a spherical soma. Arbor represents the spherical soma as a cylinder with
length and diameter equal to the diameter of the sample representing the sphere.

The Allen interpretation applies the rules above, and additional constraints imposed by the
`SONATA specification <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-biophysical-neuron-morphologies>`_.

The main difference to the standard SWC interpretation is the requirement that the soma must be represented with
a single sample with tag 1, that defines the center and radius of a spherical soma.
The SONATA specification does not explicitly state how the soma ought to be constructe from cylinders,
however it is based on the assumption that the cell will be loaded by NEURON, which implies that the soma is
treated as a cylinder:
  * with length equal to its diameter;
  * centered around the origin (0, 0, 0);
  * composed of two segments aligned along the **x axis**: segment 0 ``prox=(-r 0 0) dist=(0, 0, 0)`` and segment 1 ``prox=(0 0 0 r) dist=(r 0 0 r)`` 

This interpretation also expects that samples have the same tag as their parent, with the exception of those
whose parent is the soma. When a sample's parent is the soma, no *segment* is created
between the 2 samples; instead there is a gap in the morphology (represented electrically as a zero-resistance wire).
Samples with the soma as a parent start new segments, that are connected to the center of the soma, at the origin.
Only axons, dendrites and apical dendrites (tags 2, 3 and 4 respectively) are allowed in this interpretation,
in addition to the spherical soma.

Finally, this interpretation translates all samples such that the soma is centered around the origin.

API
"""

* :ref:`Python <pyswc>`
* :ref:`C++ <cppswc>`
