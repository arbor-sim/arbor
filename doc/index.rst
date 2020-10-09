Arbor
=====

.. image:: https://travis-ci.org/arbor-sim/arbor.svg?branch=master
    :target: https://travis-ci.org/arbor-sim/arbor

Arbor is a high-performance library for computational neuroscience simulations with multi-compartment, morphologically-detailed cells,
from single cell models to very large networks. Arbor is written from the ground up with many-cpu and gpu architectures in mind, to
help neuroscientists effectively use contemporary and future HPC systems to meet their simulation needs.

Arbor supports NVIDIA and AMD GPUs as well as explicit vectorization on CPUs from Intel (AVX, AVX2 and AVX512) and ARM (Neon and SVE).
When coupled with low memory overheads, this makes Arbor an order of magnitude faster than the most widely-used comparable simulation software.

Arbor is open source and openly developed, and we use development practices such as unit testing, continuous integration, and validation.

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
   :caption: Arbor documentation:
   :maxdepth: 1

   install/index
   tutorial/index
   concepts/index
   python/index
   cpp/index
   hpc/index
   internals/index
