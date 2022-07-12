.. _formatcablecell:

Arbor Cable Cell
================

.. csv-table::
   :header: "Name", "File extension", "Read", "Write"

   "Arbor Cable Cell", "``acc``", "✓", "✓"

We define an s-expression format for describing :ref:`cable cells <cablecell>`.
Cable cells are constructed from three components: a :ref:`label dictionary <labels>`,
a :ref:`decoration object <cablecell-decoration>` and a :ref:`morphology <morph>`.
The cable cell *format* is constructed in the same way.

.. Note::

   Extra line breaks and indentations in the s-expressions presented below have been
   added for clarity. They are not a requirement of the format and will be treated as
   whitespace.

Label dictionary
----------------

The label dictionary stores :term:`region` and :term:`locset` s-expressions with
associated labels which can then be used to refer to the region or locset when
decorating the cable cell.

Arbor provides many useful :ref:`region expressions <labels-region-expr>` and
:ref:`locset expressions <labels-locset-expr>` which are explained in detail at the
provided links.

The components of the label dictionary are the following:

.. label:: (region-def label:string reg:region)

   This defines a ``label`` which can be used to refer to the region ``reg``.
   For example:

   .. code:: lisp

      (region-def "my_region" (branch 1))

   This expression identifies the branch with id 1 as "my_region".

.. label:: (locset-def label:string ls:locset)

   This defines a ``label`` which can be used to refer to the locset ``ls``.
   For example:

   .. code:: lisp

      (locset-def "my_locset" (location 3 0.5))

   This expression identifies the midpoint of the branch with id 3 as "my_locset".

.. label:: (iexpr-def label:string e:iexpr)

   This defines a ``label`` which can be used to refer to the iexpr ``e``.
   For example:

   .. code:: lisp

      (iexpr-def "my_iexpr" (radius 0.5))

   This expression identifies the radius iexpr with a scaling factor 0.5.


Any number of locset, region an iexpr definitions can be grouped in a label dictionary as follows:

.. label:: (label-dict [...def:region-def/locset-def/iexpr-def])

   This describes a label dictionary of zero or more region, locset and iexpr definitons.
   For example:

   .. code:: lisp

      (label-dict
        (region-def "my_soma" (tag 1))
        (locset-def "root" (root))
        (region-def "all" (all))
        (region-def "my_region" (radius-ge (region "my_soma") 1.5))
        (locset-def "terminal" (terminal))
        (iexpr-def "my_iexpr" (radius 0.5)))

Decor
-----

The decor of a cable cell describes the dynamics and properties of the cell which can be assigned on
:term:`regions <region>` or :term:`locsets <locset>`, or set as defaults on the entire cell.

This table lists all supported dynamics and properties and whether they are *placeable* (i.e. they can
be placed on one or more locations on the cell described by a locset); *paintable* (i.e. they can be set
on an entire area of the cell described by a region) or *defaultable* (i.e. they are the default settings
of the cell):

.. csv-table:: Property applicability.
   :widths: 20, 10, 10, 10

                             ,           **placeable**, **paintable**, **defaultable**
   initial membrane potential,           --,             ✓,             ✓
   axial resistivity,                    --,             ✓,             ✓
   temperature,                          --,             ✓,             ✓
   membrane capacitance,                 --,             ✓,             ✓
   ion initial internal concentration,   --,             ✓,             ✓
   ion initial external concentration,   --,             ✓,             ✓
   ion initial reversal potential,       --,             ✓,             ✓
   ion reversal potential method,        --,            --,             ✓
   density mechanism,                    --,             ✓,            --
   scaled-mechanism (density),           --,             ✓,            --
   point mechanism,                      ✓,             --,            --
   junction mechanism,                   ✓,             --,            --
   current clamp,                        ✓,             --,            --
   threshold detector,                   ✓,             --,            --

The various properties and dynamics of the decor are described as follows:

.. label:: (membrane-potential val:real)

   This describes an *initial membrane potential* object with value ``val`` (unit mV).

.. label:: (axial-resistivity val:real)

   This describes an *axial resistivity* object with value ``val`` (unit Ω·cm).

.. label:: (temperature-kelvin val:real)

   This describes a *temperature* object with value ``val`` (unit K).

.. label:: (membrane-capacitance val:real)

   This describes a *membrane capacitance* object with value ``val`` (unit F/m²).

.. label:: (ion-internal-concentration ion:string val:real)

   This describes an *initial internal concentration* object for ion ``ion`` with value ``val`` (unit mM).

.. label:: (ion-external-concentration ion:string val:real)

   This describes an *initial external concentration* object for ion ``ion`` with value ``val`` (unit mM).

.. label:: (ion-reversal-potential ion:string val:real)

   This describes an *initial reversal potential* object for ion ``ion`` with value ``val`` (unit mV).

.. label:: (mechanism name:string [...(param:string val:real)])

   This describes a (point or density) mechanism object of the mechanism called ``name``. This expression
   accepts zero or more ``(param:string val:real)`` expressions. Each of these expressions sets the value of
   parameter ``param`` to ``val``.
   For example:

   .. code:: lisp

      (mechanism "hh" ("gl" 0.5) ("el" 2))

   This expression creates an "hh" mechanism and sets the "gl" and "el" parameters of the mechanism to 0.5
   and 2 respectively (units depend on the :ref:`nmodl <formatnmodl>` mechanism).

.. label:: (ion-reversal-potential-method ion:string method:mechanism)

   This creates a *reversal potential method* (able to modify the reversal potential) of ion ``ion`` from
   mechanism ``method``.
   For example:

   .. code:: lisp

      (ion-reversal-potential-method "ca" (mechanism "nernst/ca"))

.. label:: (density method:mechanism)

   This describes a *density* mechanism whose behavior is defined by ``mechanism``.

.. label:: (scaled-mechanism p:density [...(param:string e:iexpr)])

   This describes a *density* mechanism, which is modified by scaling of individual parameters with
   inhomogeneous scaling expressions.

.. label:: (synapse method:mechanism)

   This describes a *synapse* (point) mechanism whose behavior is defined by ``mechanism``.

.. label:: (junction method:mechanism)

   This describes a *gap-junction* mechanism whose behavior is defined by ``mechanism``.

.. label:: (current-clamp (envelope-pulse delay:real duration:real amplitude:real) freq:real phase:real)

   This creates a *current clamp*. If the frequency ``freq`` (unit kHz) is zero, the current is a square
   pulse with amplitude ``amplitude`` (unit nA) starting at ``delay`` (unit ms) and lasting for ``duration``
   (unit ms). If ``freq`` is non-zero, the current is sinusoidal with amplitude ``amplitude`` and frequency
   ``freq`` from time ``delay`` and lasting for ``duration``, with phase ``phase`` (unit rad) at time zero.
   (More information about current clamps can be found :ref:`here <cablecell-stimuli>`).

.. label:: (current-clamp [...(envelope time:real amplitude:real)] freq:real phase:real)

   This creates a *current clamp* with an amplitude governed by the given envelopes (``time`` unit ms and
   ``amplitude`` unit nA). A frequency ``freq`` (unit kHz) of zero implies that the generated current simply
   follows the envelope. A non-zero ``freq`` implies the current is sinusoidal with that frequency and amplitude
   that varies according to the envelope. The ``phase`` (unit rad) is the phase of the sinusoidal current
   clamp at time zero. (More information about current clamps can be found :ref:`here <cablecell-stimuli>`).
   For example:

   .. code::

      (current-clamp (envelope (0 10) (50 10) (50 0)) 0.04 0.15)

   This expression describes a sinusoidal current with amplitude 10 nA and frequency 40 Hz and that lasts
   from t = 0 ms to t = 50 ms, finally leaving the current at 0 nA (final amplitude in the envelope).

.. label:: (threshold-detector val:real).

   This describes a *threshold-detector* object with value ``val`` (unit mV).

*Paintable* and *placeable* properties and dynamics are placed on regions (generated from :ref:`region expressions
<labels-region-expr>`) and locsets (generated from :ref:`locset expressions <labels-locset-expr>`) respectively.
*Defaultable* properties and dynamics apply to an entire cell.

.. label:: (paint reg:region prop:paintable)

   This applies the painatble property ``prop`` to region ``reg``.
   For example:

   .. code:: lisp

      (paint (tag 1) (membrane-capacitance 0.02))

   This expression sets the membrane capacitance of the region tagged ``1`` to 0.02 F/m².


.. label:: (place ls:locset prop:placeable label:string)

   This places the property ``prop`` on locset ``ls`` and labels the group of items on the
   locset with ``label``. For example:

   .. code:: lisp

      (place (locset "mylocset") (threshold-detector 10) "mydetectors")

   This expression places 10 mV threshold detectors on the locset labeled ``mylocset``,
   and labels the detectors "mydetectors". The definition of ``mylocset`` should be provided
   in a label dictionary associated with the decor.

   The number of detectors placed depends on the number of locations in the "mylocset" locset.
   The placed detectors can be referred to (in the recipe for example) using the label
   "mydetectors".

.. label:: (default prop:defaultable)

   This sets the property ``prop`` as default for the entire cell. (This default property can be overridden on region
   using a ``paint`` expression).
   For example:

   .. code:: lisp

      (default (membrane-potential -65))

   This expression sets the default membrane potential of the cell to -65 mV.

Any number of paint, place and default expressions can be used to create a decor as follows:

.. label:: (decor [...def:paint/place/default])

   This describes a decor object with zero or more paint, place or default expressions in any order.
   For example:

   .. code:: lisp

      (decor
        (default (membrane-potential -55.000000))
        (paint (region "custom") (temperature-kelvin 270))
        (paint (region "soma") (membrane-potential -50.000000))
        (paint (all) (density (mechanism "pas")))
        (paint (tag 4) (density (mechanism "Ih" ("gbar" 0.001))))
        (place (locset "root") (synapse (mechanism "expsyn")) "root_synapse")
        (place (terminal) (junction (mechanism "gj")) "terminal_gj"))

Morphology
----------

The morphology of a cable cell can be described in terms of points, tagged segments and branches.

.. label:: (point x:real y:real z:real radius:real)

   This describes a 3D *point* in space with ``x``, ``y``, and ``z`` coordinates and a radius ``r`` (unit µm).

.. label:: (segment id:int prox:point dist:point tag:int)

   This describes a tapered segment from point ``prox`` to point ``dist`` with a tag ``tag`` and id ``id``.
   For example:

   .. code:: lisp

      (segment 3 (point 0 0 0 5) (point 0 0 10 2) 1)

   This expression creates a segment with id 3, with a radius that tapers linearly from 5 to 2 µm, which has a
   a tag of 1.

.. label:: (branch id:int parent_id:int seg:segment [...seg:segment])

   This describes a branch with a given ``id`` which has as a parent the branch with id ``parent_id`` (a
   ``parent_id`` equal to -1 means the branch is at the root of the morphology). The branch is composed of 1 or
   more contiguous segments ``seg``.


.. label:: (morphology [...b:branch])

   This creates the morphology from a set of branches. There exists more than one valid s-expression to
   describe the same morphology.

   For example, the shown morphology can be represented using the following s-expression. If we change
   any of the branch or segment ids, we would obtain an identical morphology.

   .. figure:: ../gen-images/label_morph.svg
     :width: 600
     :align: center

     On the left the morphology visualized using its segments, on the right using its branches.
     Python code to generate this cable cell is in the :class:`segment_tree<arbor.segment_tree>`
     documentation :ref:`here <morph-label-seg-code>`.

   .. code:: lisp

      (morphology
        (branch 0 -1
          (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1)
          (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)
          (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3))
        (branch 1 0
          (segment 3 (point 12 -0.5 0 0.8) (point 20 4 0 0.4) 3)
          (segment 4 (point 20 4 0 0.4) (point 26 6 0 0.2) 3))
        (branch 2 0
          (segment 5 (point 12 -0.5 0 0.5) (point 19 -3 0 0.5) 3))
        (branch 3 2
          (segment 6 (point 19 -3 0 0.5) (point 24 -7 0 0.2) 3))
        (branch 4 2
          (segment 7 (point 19 -3 0 0.5) (point 23 -1 0 0.2) 3)
          (segment 8 (point 23 -1 0 0.3) (point 26 -2 0 0.2) 3))
        (branch 5 -1
          (segment 9 (point 0 0 0 2) (point -7 0 0 0.4) 2)
          (segment 10 (point -7 0 0 0.4) (point -10 0 0 0.4) 2)))

Cable cell
----------

The entire cable-cell can then be constructed given the 3 previously described component
expressions.

.. label:: (cable-cell morph:morphology dec:decor dict:label-dict)

   The arguments of the cable-cell can be in any order, as long as all 3 components are listed.
   For example:

   .. code:: lisp

      (cable-cell
        (label-dict
          (region-def "my_soma" (tag 1))
          (locset-def "root" (root))
          (region-def "all" (all))
          (region-def "my_region" (radius-ge (region "my_soma") 1.5))
          (locset-def "terminal" (terminal)))
        (decor
          (default (membrane-potential -55.000000))
          (paint (region "my_soma") (temperature-kelvin 270))
          (paint (region "my_region") (membrane-potential -50.000000))
          (paint (tag 4) (density (mechanism "Ih" ("gbar" 0.001))))
          (place (locset "root") (synapse (mechanism "expsyn")) "root_synapse")
          (place (location 1 0.2) (junction (mechanism "gj")) "terminal_gj"))
        (morphology
          (branch 0 -1
            (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1)
            (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)
            (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3))
          (branch 1 0
            (segment 3 (point 12 -0.5 0 0.8) (point 20 4 0 0.4) 3)
            (segment 4 (point 20 4 0 0.4) (point 26 6 0 0.2) 3))
          (branch 2 0
            (segment 5 (point 12 -0.5 0 0.5) (point 19 -3 0 0.5) 3))
          (branch 3 2
            (segment 6 (point 19 -3 0 0.5) (point 24 -7 0 0.2) 3))
          (branch 4 2
            (segment 7 (point 19 -3 0 0.5) (point 23 -1 0 0.2) 3)
            (segment 8 (point 23 -1 0 0.3) (point 26 -2 0 0.2) 3))
          (branch 5 -1
            (segment 9 (point 0 0 0 2) (point -7 0 0 0.4) 2)
            (segment 10 (point -7 0 0 0.4) (point -10 0 0 0.4) 2))))

   This expression uses the *label-dictionary* in the *decoration* specification
   to get the descriptions of regions and locsets specified using labels.
   The *decor* is then applied on the provided *morphology*, creating a cable cell.

Parsable arbor-components and meta-data
---------------------------------------

The formats described above can be used to generate a :ref:`label dictionary <labels>`,
:ref:`decoration <cablecell-decoration>`, :ref:`morphology <morph>`, or :ref:`cable cell <cablecell>`
object. These are denoted as arbor-components. Arbor-components need to be accompanied by *meta-data*
specifying the version of the format being used. The only version currently supported is ``0.1-dev``.

.. label:: (version val:string)

   Specifies that the version of the component description format is ``val``.

.. label:: (meta-data v:version)

   Add the version information ``v`` to the meta-data of the described component.

.. label:: (arbor-component data:meta-data comp:decor/label-dict/morphology/cable-cell)

   Associates the component ``comp`` with meta-data ``data``.

The final form of each arbor-component looks as follows:

Label-dict
^^^^^^^^^^

.. code:: lisp

   (arbor-component
     (meta-data (version "0.1-dev"))
     (label-dict
       (region-def "my_soma" (tag 1))
       (locset-def "root" (root))))

Decoration
^^^^^^^^^^

.. code:: lisp

   (arbor-component
     (meta-data (version "0.1-dev"))
     (decor
       (default (membrane-potential -55.000000))
       (place (locset "root") (synapse (mechanism "expsyn")) "root_synapse")
       (paint (region "my_soma") (temperature-kelvin 270))))

Morphology
^^^^^^^^^^

.. code:: lisp

   (arbor-component
     (meta-data (version "0.1-dev"))
     (morphology
        (branch 0 -1
          (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1)
          (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)
          (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3))))

Cable-cell
^^^^^^^^^^

.. code:: lisp

   (arbor-component
     (meta-data (version "0.1-dev"))
     (cable-cell
       (label-dict
         (region-def "my_soma" (tag 1))
         (locset-def "root" (root)))
       (decor
         (default (membrane-potential -55.000000))
         (place (locset "root") (synapse (mechanism "expsyn")) "root_synapse")
         (paint (region "my_soma") (temperature-kelvin 270)))
       (morphology
          (branch 0 -1
            (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1)
            (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)
            (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3)))))

API
---

* :ref:`Python <pycablecellformat>`
* :ref:`C++ <cppcablecellformat>`
