Arbor
=====

.. image:: https://travis-ci.org/arbor-sim/arbor.svg?branch=master
    :target: https://travis-ci.org/arbor-sim/arbor

Arbor is a high-performance library for computational neuroscience simulations
with morphologically-detailed cells, from single cell models to very large networks.

The development team is from HPC centers, aiming to help neuroscientists
effectively use contemporary and future HPC systems to meet their simulation needs.

Arbor is designed from the ground up for **many core**  architectures:

    * Written in modern C++ and CUDA;
    * Distributed parallelism using MPI;
    * Multithreading with TBB and C++11 threads;
    * **Open source** and **open development**;
    * Sound development practices: **unit testing**, **continuous Integration**,
      and **validation**.

Citing Arbor
------------

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
   :caption: Concepts:

   morphology
   labels
   cable_cell
   mechanisms

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
   cpp_neuroml

.. toctree::
   :caption: Developers:

   library
   nmodl
   simd_api
   profiler
   sampling_api
   cpp_distributed_context
   cpp_dry_run

