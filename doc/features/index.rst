.. _features-overview:

Features
========

Arbor is fully featured, ready for real world modelling, and the user community is busy collaborating with us to build complex models. Best practices, tips, tricks and howtos will be documented here as the community gains experience.

Features include:

* :ref:`in_gui`: a fully featured GUI for single cell construction.
* :ref:`Support for a range of fileformats <format-overview>`, including :ref:`formatnmodl` and NeuroML by way of `NMLCC <https://github.com/arbor-sim/nmlcc>`_, a fully featured NeuroML converter for Arbor targets.
* `Nascent SONATA format support <https://github.com/arbor-sim/arbor-sonata>`_. The Allen mechanism database :ref:`is already included in Arbor <mechanisms_builtins>`, so currently, with some manual work executing Allen models is already possible, see :ref:`tutorialsinglecellallen`.
* :ref:`Voltage Processes <formatnmodl_voltageproc>` in the form of a mechanism kind. Useful for e.g. implementing voltage clamps.
* :ref:`Stochastic Differential Equations <tutorial_calcium_stpd_curve>`: both point and density mechanisms may now use white noise as part of the state updates, turning the ODEs effectively into SDEs.
* :ref:`Mutable connection table <interconnectivity-mut>`: the connection table can be redefined between successive simulation function calls.
* :ref:`Edit morphologies <morph-edit>` using Arbor API, useful for e.g. merging segment trees.
* `BluePyOpt integration <https://github.com/BlueBrain/BluePyOpt/releases/tag/1.14.0>`_. Parameter optimization using BluePyOpt can be done through an Arbor backend and exported to an Arbor cable cell, see :ref:`this tutorial <tutorialsinglecellbluepyopt>`.
* :ref:`Wide range of cable cell probes<pycablecell-probesample-api>`, :ref:`plus LIF cell probes <pycablecell-probesample-lif>`.
* :ref:`Inhomogeneous parameters <cablecell-scaled-mechs>`: enables :ref:`painting of scaled density mechanisms <labels-iexpr>`, e.g. with the distance from the root. (nmodl: ``iexpr``.)
* :ref:`LFPyKit integration <tutorial_lfpykit>`: extract extracellular signals from an Arbor cable cell.
* :ref:`Faster NMODL programming guide <formatnmodl-faster>`: helps users write NMODL with performance and fewer bugs in mind.
* :ref:`Axial Diffusion <cablecell-ions-diffusion>`: ions can now propagate along the dendrite by diffusion and be received by other synapses, modifying their weight upon reception.Xd can be read from and written to by NMODL density and point mechanisms. See `this LaTeX file <https://github.com/arbor-sim/arbor/blob/master/doc/dev/axial-diff.tex>`_ for a mathematical description of Arbor's implementation.
* :ref:`Gap Junction Mechanisms <mechanisms-junction>`.
* :ref:`Mechanism ABI <extending-catalogues>`, allowing for users to package their mechanism catalogues.
* :ref:`Built-in profiler <pyprofiler>`, which enables users to quickly understand where their experiment is spending most of its time.

Modelling
---------

Most of the Arbor community lives in our `Gitter channel <https://gitter.im/arbor-sim/community>`_\, and modellers convene `weekly <https://arbor-sim.org/arbor-weekly-videochat/>`_ to discuss how to use or identify missing features in a vide chat. Please join!

Under the `arbor-contrib <https://github.com/arbor-contrib/>`_ organisation, some users have shared their models. You can peruse these at your leisure, and of course add yours if you like to share! `Please contact us <https://docs.arbor-sim.org/en/stable/contrib/index.html#get-in-touch>`_ to have your model added to our list.
