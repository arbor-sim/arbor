Arbor
=====

|testbadge| |zlatest|

.. |testbadge| image:: https://github.com/arbor-sim/arbor/actions/workflows/basic.yml/badge.svg
    :target: https://github.com/arbor-sim/arbor/actions/workflows/basic.yml

Welcome to the documentation for Arbor, the multi-compartment neural network simulation library.

You can find out how to :ref:`get Arbor<in_install>`; get started quickly with our :ref:`tutorials<tutorial>`; or continue reading to learn more about Arbor.

What is Arbor?
--------------

`Arbor <https://arbor-sim.org>`_. is a high-performance library for computational neuroscience simulations with multi-compartment, morphologically-detailed cells,
from single cell models to very large networks. Arbor is written from the ground up with many-cpu and gpu architectures in mind, to
help neuroscientists effectively use contemporary and future HPC systems to meet their simulation needs.

Arbor supports NVIDIA and AMD GPUs as well as explicit vectorization on CPUs from Intel (AVX, AVX2 and AVX512) and ARM (Neon and SVE).
When coupled with low memory overheads, this makes Arbor an order of magnitude faster than the most widely-used comparable simulation software.

Arbor is open source and openly developed, and we use development practices such as unit testing, continuous integration, and validation.

Documentation organisation
--------------------------

* :ref:`tutorial` contains a few ready-made examples you can use to quickly get started using Arbor. In the tutorial descriptions we link to the relevant Arbor concepts.
* :ref:`modelintro` describes the design and concepts used in Arbor. The breakdown of concepts is mirrored (as much as possible) in the :ref:`pyoverview` and :ref:`cppoverview`, so you can easily switch between languages and concepts.
* The API section details our :ref:`pyoverview` and :ref:`cppoverview` API. :ref:`internals-overview` describes Arbor code that is not user-facing; convenience classes, architecture abstractions, etc.
* Contributions to Arbor are very welcome! Under :ref:`contribindex` describe conventions and procedures for all kinds of contributions.

Citing Arbor
------------

The Arbor software can be cited by version via Zenodo or via Arbors introductory paper.

Latest version
    |zlatest|

Version 0.5
    |z05|

    .. code-block:: latex

        @software{nora_abi_akar_2021_4428108,
        author       = {Nora {Abi Akar} and
                        John Biddiscombe and
                        Benjamin Cumming and
                        Felix Huber and
                        Marko Kabic and
                        Vasileios Karakasis and
                        Wouter Klijn and
                        Anne Küsters and
                        Alexander Peyser and
                        Stuart Yates and
                        Thorsten Hater and
                        Brent Huisman and
                        Sebastian Schmitt},
        title        = {arbor-sim/arbor: Arbor Library v0.5},
        month        = jan,
        year         = 2021,
        publisher    = {Zenodo},
        version      = {v0.5},
        doi          = {10.5281/zenodo.4428108},
        url          = {https://doi.org/10.5281/zenodo.4428108}
        }

Version 0.2
    |z02|

Version 0.1
    |z01|

Introductory paper
    .. code-block:: latex

        @INPROCEEDINGS{
            paper:arbor2019,
            author={N. {Abi Akar} and B. {Cumming} and V. {Karakasis} and A. {Küsters} and W. {Klijn} and A. {Peyser} and S. {Yates}},
            booktitle={2019 27th Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP)},
            title={{Arbor --- A Morphologically-Detailed Neural Network Simulation Library for Contemporary High-Performance Computing Architectures}},
            year={2019}, month={feb}, volume={}, number={},
            pages={274--282},
            doi={10.1109/EMPDP.2019.8671560},
            ISSN={2377-5750}}

    Alternative citation formats for the paper can be `downloaded here <https://ieeexplore.ieee.org/abstract/document/8671560>`_, and a preprint is available at `arXiv <https://arxiv.org/abs/1901.07454>`_.


.. |zlatest| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1459678.svg
    :target: https://doi.org/10.5281/zenodo.1459678

.. |z05| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4428108.svg
    :target: https://doi.org/10.5281/zenodo.4428108

.. |z02| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2583709.svg
    :target: https://doi.org/10.5281/zenodo.2583709

.. |z01| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1459679.svg
    :target: https://doi.org/10.5281/zenodo.1459679

Acknowledgements
----------------

This research has received funding from the European Unions Horizon 2020 Framework Programme for Research and
Innovation under the Specific Grant Agreement No. 720270 (Human Brain Project SGA1), Specific Grant Agreement
No. 785907 (Human Brain Project SGA2), and Specific Grant Agreement No. 945539 (Human Brain Project SGA3).

Arbor is an `eBrains project <https://ebrains.eu/service/arbor/>`_.

A full list of our software attributions can be found `here <https://github.com/arbor-sim/arbor/blob/master/ATTRIBUTIONS.md>`_.

.. toctree::
   :caption: Get started:
   :maxdepth: 1

   install/index
   tutorial/index

.. toctree::
   :caption: Concepts:
   :maxdepth: 1

   concepts/index
   concepts/recipe
   concepts/cell
   concepts/interconnectivity
   concepts/hardware
   concepts/domdec
   concepts/simulation
   concepts/cable_cell
   concepts/lif_cell
   concepts/spike_source_cell
   concepts/benchmark_cell

.. toctree::
   :caption: File formats:
   :maxdepth: 1

   fileformat/swc
   fileformat/neuroml
   fileformat/asc
   fileformat/nmodl
   fileformat/cable_cell

.. toctree::
   :caption: API reference:
   :maxdepth: 1

   python/index
   cpp/index
   internals/index

.. toctree::
   :caption: Contributing:
   :maxdepth: 1

   contrib/index
   contrib/pr
   contrib/coding-style
   contrib/doc
   contrib/example
   contrib/test

.. meta::
   :google-site-verification: KbkW8d9MLsBFZz8Ry0tfcQRkHsgxzkECCahcyRSjWDo
