Arbor
=====

.. image:: https://travis-ci.org/arbor-sim/arbor.svg?branch=master
    :target: https://travis-ci.org/arbor-sim/arbor

Arbor is a high-performance library for computational neuroscience simulations with multi-compartment, morphologically-detailed cells, from single cell models to very large networks. Arbor is written from the ground up with many-cpu and gpu architectures in mind, to help neuroscientists effectively use contemporary and future HPC systems to meet their simulation needs. The performance portability is by virtue of back-end specific optimizations for x86 multicore, Intel KNL, and NVIDIA GPUs. When coupled with low memory overheads, these optimizations make Arbor an order of magnitude faster than the most widely-used comparable simulation software. Arbor is open source and openly developed, and we use development practices such as unit testing, continuous integration, and validation.

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
   :caption: Getting Started:

   gs_install
   gs_python
   gs_quick_start

.. toctree::
   :caption: How does Arbor work?

   co_overview
   co_recipe
   co_cell
   co_cable_cell
   co_morphology
   co_labels
   co_mechanisms
   co_synapses
   co_hardware
   co_domdec
   co_simulation

.. toctree::
   :caption: Python API

   py_overview
   py_recipe
   py_cell
   py_cable_cell
   py_morphology
   py_labels
   py_mechanisms
   py_synapses
   py_hardware
   py_domdec
   py_simulation
   py_profiler
   py_reference

.. toctree::
   :caption: C++ API

   cpp_overview
   cpp_recipe
   cpp_cell
   cpp_cable_cell
   cpp_synapses
   cpp_hardware
   cpp_domdec
   cpp_simulation
   cpp_profiler
   cpp/reference

.. toctree::
   :caption: C++ API for HPC

   cpp_distributed_context
   cpp_dry_run

.. toctree::
   :caption: Arbor Internals

   ai_library
   ai_nmodl
   ai_simd_api
   ai_sampling_api

