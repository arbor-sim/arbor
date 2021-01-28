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
   initial membrane potential                init-membrane-potential    mV
   temperature                               temperature-K              kelvin
   axial resistivity                         axial-resistivity          Ω·cm
   membrane capacitance                      membrane-capacitance       F⋅m⁻²
   initial internal concentration (per ion)  init-int-concentration     mM
   initial external concentration (per ion)  init-ext-concentration     mM
   initial reversal potential (per ion)      init-reversal-potential    mV
   reversal potential method (per ion)       reversal-potential-method  --
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
         "init-int-concentration": 5e-05,
         "init-ext-concentration": 2.0,
         "init-reversal-potential": 132.4579341637009
       },
       "k": {
         "init-int-concentration": 54.4,
         "init-ext-concentration": 2.5,
         "init-reversal-potential": -77.0,
       },
       "na": {
         "init-int-concentration": 10.0,
         "init-ext-concentration": 140.0,
         "init-reversal-potential": 50.0,
         "reversal-potential-method": {
           "mechanism" :"nernst/na"
         }
       }
     }
   }

API
~~~

* :ref:`Python <pyjsonformats>`
* :ref:`C++ <cppjsonformats>`
