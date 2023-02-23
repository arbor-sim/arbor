.. _pylabels:

Cable cell labels
=================

.. currentmodule:: arbor

.. py:class:: label_dict

   Stores labels and their associated :ref:`expressions <labels-expressions>` as key-value pairs.

   .. method:: label_dict()

      Create an empty label dictionary

   .. method:: label_dict(dictionary)
      :noindex:

      Initialize a label dictionary from a ``dictionary`` with string labels as keys,
      and corresponding string definitions as values.

      .. code-block:: python

         labels = arbor.label_dict({'soma':   '(tag 1)',           # region
                                    'midpoint': '(location 0 0.5)'}) # locset

   .. attribute:: regions

      The region definitons in the dictionary.

   .. attribute:: locsets

      The locset definitons in the dictionary.

The ``arbor.label_dict`` type is used for creating and manipulating label dictionaries,
which can be initialised with a dictionary that defines (label, :ref:`expression <labels-expressions>`)
pairs. For example, a dictionary that uses tags that correspond to SWC
`structure identifiers <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_
to label soma, axon, dendrite and apical dendrites is:

.. code-block:: python

    import arbor

    labels = {'soma': '(tag 1)',
              'axon': '(tag 2)',
              'dend': '(tag 3)',
              'apic': '(tag 4)'}

    d = arbor.label_dict(labels)

The same ``label_dict`` can be created by starting with an empty label dictionary and adding the labels and their definitions one by one:

.. code-block:: python

    import arbor

    d = arbor.label_dict()

    d['soma'] = '(tag 1)'
    d['axon'] = '(tag 2)'
    d['dend'] = '(tag 3)'
    d['apic'] = '(tag 4)'

The square bracket operator is used above to add label definitions. It can
be used to modify existing definitions, so long as the new new definition has the
same type (region or locset):

.. code-block:: python

    import arbor

    # A label dictionary that defines the label "dend" that defines a region.
    d = arbor.label_dict({'dend': '(tag 3)')

    # The definition of a label can be overwritten with a definition of the
    # same type, in this case a region.
    d['dend'] = '(join (tag 3) (tag 4))'

    # However, a region can't be overwritten by a locset, or vice-versa.
    d['dend'] = '(terminal)' # error: '(terminal)' defines a locset.

    # New labels can be added to the dictionary.
    d['soma'] = '(tag 1)'
    d['axon'] = '(tag 2)'

    # Square brackets can also be used to get a label's definition.
    assert(d['soma'] == '(tag 1)')

:Ref:`Expressions <labels-expressions>` can refer to other regions and locsets in a label dictionary.
In the example below, we define a region labeled *'tree'* that is the union
of both the *'dend'* and *'apic'* regions.

.. code-block:: python

    import arbor

    d = arbor.label_dict({
            'soma': '(tag 1)',
            'axon': '(tag 2)',
            'dend': '(tag 3)',
            'apic': '(tag 4)',
            # equivalent to (join (tag 3) (tag 4))
            'tree': '(join (region "dend") (region "apic"))'})

The order that labels are defined does not matter, so an :ref:`expression <labels-expressions>` can refer to a
label that has not yet been defined:

.. code-block:: python

    import arbor

    d = arbor.label_dict()
    # 'reg' refers
    d['reg'] = '(distal_interval (locset "loc"))'
    d['loc'] = '(location 3 0.5)'

    # If d was applied to a morphology, 'reg' would refer to the region:
    #   '(distal_interval (location 3 0.5))'
    # Which is the sub-tree of the matrix starting at '(location 3 0.5)'

    # The locset 'loc' can be redefined
    d['loc'] = '(proximal (tag 3))'

    # Now if d was applied to a morphology, 'reg' would refer to:
    #   '(distal_interval (proximal (tag 3))'
    # Which is the subtrees that start at the proximal locations of
    # the region '(tag 3)'

Cyclic dependencies are not permitted, as in the following example where
two labels refer to one another:

.. code-block:: python

    import arbor

    d = arbor.label_dict()
    d['reg'] = '(distal_interval (locset "loc"))'
    d['loc'] = '(proximal (region "reg"))'

    # Error: 'reg' needs the definition of 'loc', which in turn needs the
    # definition of 'reg'.

.. note::
    In the example above there will be no error when the label dictionary is defined.
    Instead, there will be an error later when the label dictionary is applied to
    a morphology, and the cyclic dependency is detected when thingifying the locations
    in the locsets and the cable segments in the regions.


The type of an :ref:`expression <labels-expressions>`, locset or region, is inferred automatically when it is
input into a label dictionary.
Lists of the labels for regions and locsets are available as attributes:

.. code-block:: python

    import arbor

    d = arbor.label_dict({
            'soma': '(tag 1)',
            'axon': '(tag 2)',
            'dend': '(tag 3)',
            'apic': '(tag 4)',
            'site': '(location 2 0.5)',
            'term': '(terminal)'})
