.. _tutorialfullfeaturediffusion:

A Diffusion Process with Full Feedback
======================================

Originally we intended for the axial diffusion process to be fully integrated
into the cable model. However, for various technical reasons, this has proven
infeasible and was replaced by introduction of an extra variable ``Xd`` which
is subject to diffusion as opposed to the internal concentration.

This tutorial will show you how to build a simulation that fully integrates
diffusive ion dynamics with a cable cell neuron. While we focus on a single
neuron, this should simply work as is in a network of cells.

Theory
------

In order to build a complete model, we will need to review the foundational
equations as implemented in Arbor. We assume a single ion species :math:`s` with
valence :math:`q_s` and no leak currents such that :math:`I_m = I_s`. More ion
species can be added, but do not change the fundamental mechanisms. First, the
cable equation

.. math::

   \frac{1}{C}\dot U_m = \sigma \Delta U_m + I_m

where

.. math::

   I_s = g_s (U_m - E_s)

with the reversal potential :math:`E_s`. The current :math:`I_s` -- or actually
the conductance :math:`g_s` -- is usually computed by a density mechanism.

Note that we use a slight simplifications here; :math:`\sigma` and the cable
radius :math:`r` are taken as constant along the dendrite. Second, the diffusion
equation under similar assumptions

.. math::

   \dot X_s = \beta \Delta X_s

where :math:`X_s` is the diffusive concentration of species :math:`s`. Finally,
the Nernst equation

.. math::

   E_s = \frac{R\cdot T}{q_s\cdot F}\log\left(\frac{X_o}{X_i}\right)

where :math:`X_{i,o}` are the internal/external concentrations of species
:math:`s` If your model does not use the Nernst equation to compute :math:`E_s`,
you can ignore this part, but be aware that this approach will not have a closed
feedback loop.

Incorporating Diffusion
-----------------------

We will leave the cable equation and the current computation unaltered. However,
from the intuition that a current is caused by movement of ions across the membrane,
one would expect :math:`I_s` to exact a change in :math:`X_s`. Thus, we will add an
extra term to the diffusion equation

.. math::

   \dot X_s = \beta \Delta X_s + \frac{I_s}{q_s\cdot F\cdot V}

where Faraday's constant :math:`F` scales from current to change in molar
amount, in conjunction with the charge per ion :math:`q_s`, which we scale again
by the volume :math:`V` to arrive at a change in concentration.

We implement this as an NMODL file intended to be added to the full cell that
should look like this

.. code-block::

    NEURON {
        SUFFIX current_to_delta_x
        USEION x READ ix WRITE xd, xi VALENCE zx
        GLOBAL F
        RANGE xi0
    }

    PARAMETER {
        F = 96485.3321233100184 (coulomb/mole) : Faraday's constant
        diam
        xi0                                    : initial concentration
    }

    STATE { xi }

    INITIAL {
        xi = xi0
        xd = xi0
    }

    BREAKPOINT  {
        : area   = pi * length * radius
        : volume = pi * length * radius^2
        : thus area/volume = radius
        xd = xd + 2*ix/(q*F*diam)
        xi = xd
    }

where we exploit our knowledge about the cylindrical geometry of the CVs to
compute :math:`\frac{A}{V} = \frac{1}{r}`. This mechanism also turns the
internal concentration ``xi`` into a ``STATE`` variable, which is the standard
way of handling concentration mechanisms, and couples it to ``xd`` directly.
Note that this requires explicit intialisation of both ``xi`` and ``xd``.

Setting up a Simulation
-----------------------

Having layed the foundation, adding this to a simulation is pretty simple. Save
the NMODL file, add it to your local catalogue, and compile & load that via the
usual method.

First, declare the ion -- we'll use a new species ``X`` here, but any name will
do -- by calling ``set_ion("X", valence=1)`` on your global properties object.
This object is part of the ``recipe`` or ``single_cell_model``, the use of both
is explained in other tutorials. When using an existing ion, this upfront
declaration is not needed and you'll get the default values for this ion.

.. code-block:: python

   dec = (A.decor()
       # Add our new ion to the cell; the `int_con` value has no effect.
       .set_ion("X", int_con=0.0, ext_con=42.0, diff=0.005, method="nernst/X") )
       # Place a synapse that _directly_ adds to the diffusive concentration
       .place("(location 0 0.5)", A.synapse("inject/x=X", {"alpha": 200.0}), "Zap")
       # also add an exponential decay to Xd
       .paint("(all)", A.density("decay/x=X"))
       # turn iX into a change in Xd and bind Xi to Xd
       .paint("(all)", A.density("current_to_delta_x/x=X", {"xi0": 10.0}))
       # ...
   )

While simple, note some subtleties around our custom concentration mechanism:
- The mechanism ``current_to_delta_x`` uses ``xi`` as a ``STATE`` and is thus
  solely responsible for managing its value. This makes adding an explicit
  initialisation via ``xi0`` necessary. Only one mechanism with this property
  should exist. See above for an alternative.
- The change in ``xd`` due to events arriving at the synapse ``Zap`` will be
  synchronised with ``xi`` in our custom mechanism. If no concentration
  mechanism is used, the synapse needs to be modified to write to ``xi`` as well.
- By using ``xi=xd``, the Nernst mechanism will pick up the correct value for
  ``xi``. If that is not your intention, you will have to provide a modified
  version of ``nernst`` in which ``xi`` is replaced with ``xd``.

Conclusion
----------

Apart from some theory, adding an ion with diffusion and full feedback via the
transmembrane current to a simulation is actually quite straightforward. You
might also consider changing the external concentration ``Xo`` according to the
ion current ``iX``. This was not shown above for two reasons. First, Arbor does
not handle extra-cellular dynamics and thus has no extra-cellular diffusion.
Second, the method for handling this is identical to what we have done for
``xi``, so including it doesn't add any insight.
