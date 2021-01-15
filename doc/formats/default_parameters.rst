.. _formatsdefault:

Default parameters
------------------

Arbor supports reading default model parameters of cable-cells from JSON.
These `default parameters` are applied at the global level of a model, affecting
all simulated cable-cells. Some of these defaults can later be overridden at the
cell level, which can also be done using JSON (see
:ref:`decor formats <formatsdecor>` for more details.)

The `default parameters` of a model are:

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
   CV policy                                 --                         --
   ========================================  =========================  =========

Of all possible default parameters, all but the CV policy can be specified using the
JSON format.

The following parameters are mandatory and must be set by the user:

  * The initial membrane potential
  * The temperature
  * The axial resistivity
  * The membrane capacitance
  * The initial internal and external concentrations of the 'ca', 'na' and 'k' ions.
  * The initial reversal potential of the 'ca', 'na' and 'k' ions.

The reversal potential method is optional. It can be:

  * "constant"  (the reversal potential of an ion doesn't change during the simulation.)
  * "nernst"    (the reversal potential of an ion advances accoridng to the nernst equation.)
  * any other string that represents the name of a valid :ref:`nmodl mechanism <mechanisms-revpot>`
    which describes the progression of the reversal potential.

If the reversal potential method is not set by the user, it is automatically set to
"constant".

The JSON format has a structure similar to the following example:

.. code:: JSON

   {
     "Ra": 35.4,
     "Vm": -60.0,
     "celsius": 6.3,
     "cm": 0.01,
     "ions": {
       "ca": {
         "internal-concentration": 5e-05,
         "external-concentration": 2.0,
         "reversal-potential": 132.4579341637009
       },
       "k": {
         "internal-concentration": 54.4,
         "external-concentration": 2.5,
         "reversal-potential": -77.0,
         "method": "constant"
       },
       "na": {
         "internal-concentration": 10.0,
         "external-concentration": 140.0,
         "reversal-potential": 50.0,
         "method": "nernst"
       }
     }
   }

API
~~~

* :ref:`Python <pyjsonformats>`
* :ref:`C++ <cppjsonformats>`
