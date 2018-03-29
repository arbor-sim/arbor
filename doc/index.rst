Arbor
=====

.. image:: https://travis-ci.org/eth-cscs/arbor.svg?branch=master
    :target: https://travis-ci.org/eth-cscs/arbor

What is Arbor?
--------------

Arbor is a high-performance library for computational neurscience simulations.

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

We are actively developing `Arbor <https://github.com/eth-cscs/arbor>`_, improving performance and adding features.
Some key features include:

    * Optimized back ends for CUDA, KNL and AVX2 intrinsics.
    * Asynchronous spike exchange that overlaps compute and communication.
    * Efficient sampling of voltage and current on all back ends.
    * Efficient implementation of all features on GPU.
    * Reporting of memory and energy consumption (when available on platform).
    * An API for addition of new cell types, e.g. LIF and Poisson spike generators.
    * Validation tests against numeric/analytic models and NEURON.

.. toctree::
   :caption: Getting Stared:

   install

.. toctree::
   :caption: Users:

   users

.. toctree::
   :caption: C++ API:
   :glob:

   cpp_intro
   cpp_common
   cpp_recipe

.. toctree::
   :caption: Developers:

   library
   profiler
   sampling_api

