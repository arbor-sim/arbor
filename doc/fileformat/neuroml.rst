.. _formatneuroml:

NeuroML2
--------

.. csv-table::
   :header: "Name", "File extension", "Read", "Write"

   "NeuroML2", "``nml``", "✓", "✗"

Arbor offers limited support for models described in `NeuroML version 2
<https://neuroml.org/neuromlv2>`_. Arbor is able to parse and check the validity
of morphologies described in NeuroML files, and present the encoded data to the
user. This is more than a simple `segment tree`.

NeuroML can encode in the same file multiple top-level morphologies, as well as cells:

Example
^^^^^^^

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

The above NeuroML description defines two top-level morphologies ``m1`` and ``m2`` (empty); a cell ``c1`` that uses
morphology ``m1``; and a cell ``c2`` that uses an internally defined (empty) morphology ``m3``.

Arbor can query the cells and morphologies using their ids and return all the associated morphological data for each.
The morphological data includes the actual morphology as well as the named segments and groups of the morphology.
For example, the above ``m1`` morphology has one named segment ``seg-0`` and one named group ``group-0`` that are
both represented using Arbor's :ref:`region expressions <labels-expressions>`.

API
^^^

* :ref:`Python <pyneuroml>`
* :ref:`C++ <cppneuroml>`
