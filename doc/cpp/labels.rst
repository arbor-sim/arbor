.. _cpplabels:

Cable cell labels
=================

.. currentmodule:: arbor

.. cpp:class:: label_dict

   Stores labels and their associated :ref:`expressions <labels-expressions>` as key-value pairs.

   .. cpp:function:: label_dict()

   .. cpp:function:: label_dict(const label_dict&)

   .. cpp:function:: label_dict& extend(const label_dict& other, const std::string& prefix = "")

      Add all definitions from ``other``, optionally adding a prefix.
          
   .. cpp:function:: const std::unordered_map<std::string, arb::region>& regions() const

      The region definitions in the dictionary.

   .. cpp:function:: const std::unordered_map<std::string, arb::locset>& locsets() const

      The locset definitions in the dictionary.

   .. cpp:function:: const std::unordered_map<std::string, arb::iexpr>& iexpressions() const

      The iexpr definitions in the dictionary.

   .. cpp:function:: label_dict& set(const std::string& name, locset ls)

      Add locset under ``name``.
      
   .. cpp:function:: label_dict& set(const std::string& name, region reg)

      Add region under ``name``.

   .. cpp:function:: label_dict& set(const std::string& name, iexpr e)

      Add iexpr under ``name``.

   .. cpp:function:: add_swc_tags

      Add SWC default regions

   .. cpp:function:: std::size_t erase(const std::string& key)

      Remove definitions for ``key``.
      
   .. cpp:function:: std::optional<locset> locset(const std::string& name)

      Try to find locset under ``name``.

   .. cpp:function:: std::optional<region> region(const std::string& name)

      Try to find region under ``name``.

   .. cpp:function:: std::optional<iexpr> iexpr(const std::string& name)

      Try to find iexpr under ``name``.

The ``arb::label_dict`` type is used for creating and manipulating label
dictionaries. For example, a dictionary that uses tags that correspond to SWC
`structure identifiers
<http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_
to label soma, axon, basal dendrites, and apical dendrites is:

.. code-block:: cpp

   using namespace arborio::literals;

   auto d = arb::label_dict();

   // same as d.add_swc_tags()
   d.set("soma", arb::reg::tagged(1));
   d.set("axon", arb::reg::tagged(2));
   d.set("dend", arb::reg::tagged(3));
   d.set("apic", arb::reg::tagged(4));

The ``set`` method is used above to add label definitions. It can be
used to modify existing definitions, so long as the new definition has the same
type (region or locset), continuing:

.. code-block:: cpp

    // A label can be overwritten with a definition of the
    // same type, in this case, a region.
    d.set("dend", "(join (tag 3) (tag 4))"_reg);

    // However, a region can"t be overwritten by a locset, or vice-versa.
    d.set("dend", "(terminal)"_ls); error: "(terminal)" defines a locset.

:Ref:`Expressions <labels-expressions>` can refer to other regions and locsets
in a label dictionary. In the example below, we define a new region labeled that
is the union of both the ``dend`` and ``apic`` regions.

.. code-block:: cpp

    // equivalent to (join (tag 3) (tag 4))
    d.set("3+4", arb::reg::join(arb::reg::named("dend"),
                                arb::reg::named("apic")));

The order which labels are defined in does not matter, so an :ref:`expression
<labels-expressions>` can refer to a label that has not yet been defined,
continuing our running example

.. code-block:: cpp

    // If d were applied to a morphology, "reg" would refer to the region:
    //   "(distal_interval (location 3 0.5))"
    // Which is the sub-tree of the matrix starting at "(location 3 0.5)"
    d.set("reg", "(distal_interval (locset \"loc\"))"_ls);
    d.set("loc", "(location 3 0.5)"_ls);

    // The locset "loc" can be redefined
    d.set("loc",  "(proximal (tag 3))"_ls);

    // Now if d were applied to a morphology, "reg" would refer to:
    //   "(distal_interval (proximal (tag 3))"
    // Which is the subtrees that start at the proximal locations of
    // the region "(tag 3)"

Cyclic dependencies are not permitted, as in the following example where
two labels refer to one another:

.. code-block:: cpp

    d.set("reg", "(distal_interval (locset \"loc\"))"_reg);
    d.set("loc", "(proximal (region \"reg\"))"_ls);

.. note::

   In the example above, there will be no error when the label dictionary is
   defined. Instead, there will be an error later when the label dictionary is
   applied to a morphology, and the cyclic dependency is detected when
   thingifying the locations in the locsets and the cable segments in the
   regions.
