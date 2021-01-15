.. _formatsdecor:

Decor
-----

Arbor supports reading parts of the :ref:`decor <cablecell-decoration>`
of a cable-cell from JSON. Namely, the parameters of the cell, the parameters
on regions of the cell and the mechanisms on regions of the cell.
We refer to these as the `global parameters`, the `local parameters` and
`the mechanisms` respectively.

We also refer to the `default parameters`, these are the parameters that
apply to every cell in the model, which which can also be
:ref:`read from JSON <formatsdefault>`.

The `local parameters` of a cell override the `global parameters` of a
cell, which in turn override the `default parameters` of the model.

Global parameters
~~~~~~~~~~~~~~~~~
The `global parameters` refer to the parameters that apply to the entire
cell. They contain the same elements as the `default parameters`.

   ========================================  =========================  =========
   parameter                                 alias                      units
   ========================================  =========================  =========
   initial membrane potential                Vm                         mV
   temperature                               celsius                    celsius
   axial resistivity                         Ra                         Ω·cm
   membrane capacitance                      cm                         μf⋅cm⁻²
   initial internal concentration (per ion)  internal-concentration     mM
   initial external concentration (per ion)  external-concentration     mM
   initial reversal potential (per ion)      reversal-potential         mV
   reversal potential method (per ion)       method                     --
   ========================================  =========================  =========

None of these parameters are mandatory. If unspecified, the corresponding
parameter from the `default properties` is used.

The JSON format of the `global parameters` has a structure similar to the
following example:

.. code:: JSON

   {
     "global":
      {
         "Ra": 35.4,
         "cm": 0.01,
         "ions": {
           "ca": {
             "reversal-potential": 132.4579341637009
           },
           "na": {
             "internal-concentration": 10.0
           }
         }
      }
   }

Local parameters
~~~~~~~~~~~~~~~~

The `local parameters` refer to the parameters that can be set on
:term:`regions <region>` of the cell. All of the the previously identified
`global parameters` are also `local parameters` **except** the reversal
potential method, which must be the same across the entire cell, and
therefore can't be applied to a single region.

To specify which region the local parameters must be applied to, the
corresponding expression of the region is used (ex: "(tag 1)").
Alternatively, if the region has been declared in a
:ref:`label dictionary <labels-dictionary>` associated with the same cell
as the resulting decor, the label can be used directly (ex: "soma").

The local parameters are applied to the regions in the order which they
appear in the JSON array. This is important because if the same parameter
is set to different values on the same regions or overlapping regions,
the latter value in the array will be used.

No local parameters are mandatory. If unspecified, the corresponding
parameter from the `global properties` is used.

The JSON format of the `local parameters` has a structure similar to the
following example:

.. code:: JSON

   {
     "local":
     [
       {
         "region": "(tag 4)",
         "cm": 0.02,
         "ions": {
           "na": {"reversal-potential":  50},
           "k":  {"reversal-potential": -85}
         }
       },
       {
         "region": "(tag 2)",
         "ions": {
           "na": {"reversal-potential":  50},
           "k":  {"reversal-potential": -85}
         }
       },
       {
         "region": "soma",
         "cm": 0.02
       }
     ]
   }

Mechanisms
~~~~~~~~~~

It is also possible to select which mechanisms are to be painted on which regions
and set their parameters using JSON.

Any of the mechanisms from an arbor :ref:`catalogue <mechanisms-cat>` can
be selected, provided that the catalogue is selected in the overarching model.
They are specified using the name of the mechanism, and the region is specified
either by using the expression or by using the label (similar to the
`local parameters`).

The JSON format of the `mechanisms` has a structure similar to the following
example (using mechanisms from the ``bbp`` and ``default`` catalogues):

.. code:: JSON

  {
    "mechanisms":
    [
      {
        "region": "(all)",
        "mechanism": "pas",
        "parameters": {"e": -75, "g": 3e-5}
      },
      {
        "region": "(region \"soma\")",
        "mechanism": "CaDynamics_E2",
        "parameters": {"gamma": 0.000609, "decay": 210.485284, "initCai": 5e-5}
      },
      {
        "region": "(region \"soma\")",
        "mechanism": "SKv3_1",
        "parameters": {"gSKv3_1bar": 0.303472}
      },
      {
        "region": "dend",
        "mechanism": "SK_E2",
        "parameters": {"gSK_E2bar": 0.008407}
      }
    ]
  }

Full Format
~~~~~~~~~~~

The `global parameters`, `local parameters` and `mechnaisms` can all be defined in the same json file:

.. code:: JSON

   {
     "global":
      {
         "Ra": 35.4,
         "cm": 0.01,
         "ions": {
           "ca": { "reversal-potential": 132.4579341637009 }
         }
      },
     "local":
     [
       {
         "region": "(tag 4)",
         "cm": 0.02,
         "ions": { "k":  {"reversal-potential": -85} }
       },
       {
         "region": "soma",
         "cm": 0.02
       }
     ],
     "mechanisms":
     [
       {
         "region": "(all)",
         "mechanism": "pas",
         "parameters": {"e": -75, "g": 3e-5}
       },
       {
         "region": "(region \"soma\")",
         "mechanism": "CaDynamics_E2",
         "parameters": {"gamma": 0.000609, "decay": 210.485284}
       }
     ]
   }

API
~~~

* :ref:`Python <pyjsonformats>`
* :ref:`C++ <cppjsonformats>`