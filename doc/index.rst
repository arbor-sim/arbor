Arbor
=====

.. image:: https://travis-ci.org/arbor-sim/arbor.svg?branch=master
    :target: https://travis-ci.org/arbor-sim/arbor

What is Arbor?
--------------

Arbor is a high-performance library for computational neuroscience simulations.

The development team is from from high-performance computing (HPC) centers:

    * Swiss National Supercomputing Center (CSCS), Jülich and BSC in work package 7.5.4 of the HBP.
    * Aim to prepare neuroscience users for new HPC architectures;

Arbor is designed from the ground up for **many core**  architectures:

    * Written in C++11 and CUDA;
    * Distributed parallelism using MPI;
    * Multithreading with TBB and C++11 threads;
    * **Open source** and **open development**;
    * Sound development practices: **unit testing**, **continuous Integration**,
      and **validation**.

Features
--------

We are actively developing `Arbor <https://github.com/arbor-sim/arbor>`_, improving performance and adding features.
Some key features include:

    * Optimized back end for CUDA
    * Optimized vector back ends for Intel (KNL, AVX, AVX2) and Arm (ARMv8-A NEON) intrinsics.
    * Asynchronous spike exchange that overlaps compute and communication.
    * Efficient sampling of voltage and current on all back ends.
    * Efficient implementation of all features on GPU.
    * Reporting of memory and energy consumption (when available on platform).
    * An API for addition of new cell types, e.g. LIF and Poisson spike generators.
    * Validation tests against numeric/analytic models and NEURON.

Citing Arbor
------------

.. |DOI-v0.1| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1459679.svg
     :target: https://doi.org/10.5281/zenodo.1459679

.. |DOI-v0.2| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2583709.svg
    :target: https://doi.org/10.5281/zenodo.2583709

Specific versions of Arbor can be cited via Zenodo:

   * v0.2:  |DOI-v0.2|
   * v0.1:  |DOI-v0.1|

The following BibTeX can be used to cite Arbor:

.. code-block:: latex

    @INPROCEEDINGS{
        paper:arbor2019,
        author={N. A. {Akar} and B. {Cumming} and V. {Karakasis} and A. {Küsters} and W. {Klijn} and A. {Peyser} and S. {Yates}},
        booktitle={2019 27th Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP)},
        title={{Arbor --- A Morphologically-Detailed Neural Network Simulation Library for Contemporary High-Performance Computing Architectures}},
        year={2019}, month={feb}, volume={}, number={},
        pages={274--282},
        doi={10.1109/EMPDP.2019.8671560},
        ISSN={2377-5750}}

Alternative citation formats for the paper can be `downloaded here <https://ieeexplore.ieee.org/abstract/document/8671560>`_, and a preprint is available at `arXiv <https://arxiv.org/abs/1901.07454>`_.

.. toctree::
   :caption: Getting Stared:

   install
   python
   single_cell

.. toctree::
   :caption: Arbor Models:

   model_intro
   model_concepts
   model_hardware
   model_recipe
   model_domdec
   model_simulation

.. toctree::
   :caption: Python:
   
   py_intro
   py_common
   py_recipe
   py_cable_cell
   py_hardware
   py_domdec
   py_simulation
   py_profiler

.. toctree::
   :caption: C++ API:

   cpp_intro
   cpp_common
   cpp_hardware
   cpp_recipe
   cpp_domdec
   cpp_simulation
   cpp_cable_cell

.. toctree::
   :caption: Developers:

   library
   simd_api
   profiler
   sampling_api
   cpp_distributed_context
   cpp_dry_run

