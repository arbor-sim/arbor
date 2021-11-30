.. _spikecell:

Spike source cells
==================

Spiking cells act as spike sources from user-specified values inserted via a `schedule description`.
They are typically used as stimuli in a network of more complex cells.

A spike source cell:

* has its morphology is automatically modelled as a single :term:`compartment <control volume>`;
* has one built-in **source**, which needs to be given a label to be used when forming connections from the cell;
* has no **targets**;
* does not support adding additional **sources** or **targets**;
* does not support **gap junctions**;
* does not support adding density or point mechanisms;
* can only interact with other cells via spike exchange over a :ref:`connection <modelconnections>`
  where they be a *source* of spikes to cells that have target sites
  (i.e. *cable* and *lif* cells), but they can not *receive* spikes.

API
---

* :ref:`Python <pyspikecell>`
* :ref:`C++ <cppspikecell>`
