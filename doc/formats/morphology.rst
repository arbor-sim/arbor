.. _formatsmorph:

Morphology
----------

Arbor supports morphologies described using the SWC file format and the NeuroML file format.

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

Arbor interpretation:
"""""""""""""""""""""
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


Allen interpretation:
"""""""""""""""""""""
In addition to the previously mentioned checks, the Allen interpretation expects a single-sample soma to be the first
sample of the file and to be interpreted as a spherical soma. Arbor represents the spherical soma as a cylinder with
length and diameter equal to the diameter of the sample representing the sphere.

This interpretation also expects that samples have the same tag as their parent samples, with the exception of samples
that have the soma sample as a parent. In this case, when a sample's parent is the soma, no *segment* is created
between the 2 samples; instead there is a gap in the morphology (represented electrically as a zero-resistance wire).
Samples with the soma as a parent start new segments, that connect to the distal end of the soma if they are dendrites,
or to the proximal end of the soma if they are axons or apical dendrites. Only axons, dendrites and apical dendrites
(tags 2, 3 and 4 respectively) are allowed in this interpretation, in addition to the spherical soma.

Finally the Allen institute interpretation of SWC files centres the morphology around the soma at the origin (0, 0, 0)
and all samples are translated in space towards the origin.

NEURON interpretation:
""""""""""""""""""""""
The NEURON interpretation was obtained by experimenting with the ``Import3d_SWC_read`` function. We came up with the
following set of rules that govern NEURON's SWC behavior and enforced them in arbor's NEURON-complaint SWC
interpreter:

* SWC files must contain a soma sample and it must to be the first sample.
* A soma is represented by a series of n≥1 unbranched, serially listed samples.
* A soma is constructed as a single cylinder with diameter equal to the piecewise average diameter of all the
  segments forming the soma.
* A single-sample soma at is constructed as a cylinder with length=diameter.
* If a non-soma sample is to have a soma sample as its parent, it must have the most distal sample of the soma
  as the parent.
* Every non-soma sample that has a soma sample as its parent, attaches to the created soma cylinder at its midpoint.
* If a non-soma sample has a soma sample as its parent, no segment is created between the sample and its parent,
  instead that sample is the proximal point of a new segment, and there is a gap in the morphology (represented
  electrically as a zero-resistance wire)
* To create a segment with a certain tag, that is to be attached to the soma, we need at least 2 samples with that
  tag.

NeuroML
~~~~~~~

Arbor offers limited support for models described in `NeuroML version 2 <https://neuroml.org/neuromlv2>`_.
This is not built by default (see :ref:`NeuroML support <install-neuroml>` for instructions on how
to build arbor with NeuroML).

Once support is enabled, Arbor is able to parse and check the validity of morphologies described in NeuroML files,
and present the encoded data to the user.  This is more than a simple a `segment tree`.

NeuroML can encode in the same file multiple top-level morphologies, as well as cells:

.. code:: XML

   <neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
   <morphology id="m1">
       <segment id="seg-0">
           <proximal x="1" y="1" z="1" diameter="1"/>
           <distal x="2" y="2" z="2" diameter="2"/>
       </segment>
       <segmentGroup id="group-0">
           <member segment="1"/>
       </segmentGroup>
   </morphology>
   <morphology id="m2"/>
   <cell id="c1" morphology="m1"/>
   <cell id="c2">
       <morphology id="m3"/>
   </cell>
   </neuroml>

The above NeuroML description defines 2 top-level morphologies ``m1`` and ``m2`` (empty); a cell ``c1`` that uses
morphology ``m1``; and a cell ``c2`` that uses an internally defined (empty) morphology ``m3``.

Arbor can query the cells and morphologies using their ids and return all the associated morphological data for each.
The morphological data includes the actual morphology as well as the named segments and groups of the morphology.
For example, the above ``m1`` morphology has one named segment ``seg-0`` and one named group ``group-0`` that are
both represented using Arbor's :ref:`region expressions <labels-expressions>`.

API
~~~

* :ref:`Python <pymorph-formats>`
* :ref:`C++ <cppmorphology-formats>`