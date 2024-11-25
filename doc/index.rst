Arbor
=====

|ci| |spack| |pip| |pythonwheels| |zlatest| |gitter| |matrix|

.. |ci| image:: https://github.com/arbor-sim/arbor/actions/workflows/test-matrix.yml/badge.svg
    :target: https://github.com/arbor-sim/arbor/actions/workflows/test-matrix.yml

.. |spack| image:: https://github.com/arbor-sim/arbor/actions/workflows/test-spack.yml/badge.svg
    :target: https://github.com/arbor-sim/arbor/actions/workflows/test-spack.yml

.. |pip| image:: https://github.com/arbor-sim/arbor/actions/workflows/test-pip.yml/badge.svg
    :target: https://github.com/arbor-sim/arbor/actions/workflows/test-pip.yml

.. |pythonwheels| image:: https://github.com/arbor-sim/arbor/actions/workflows/build-pip-wheels.yml/badge.svg
    :target: https://github.com/arbor-sim/arbor/actions/workflows/build-pip-wheels.yml

.. |gitter| image:: https://badges.gitter.im/arbor-sim/community.svg
    :target: https://gitter.im/arbor-sim/community

.. |matrix| image:: https://matrix.to/img/matrix-badge.svg
    :target: https://matrix.to/#/#arbor-sim_community:gitter.im

Welcome to the documentation for Arbor, the multi-compartment neural network simulation library.

You can find out how to :ref:`get Arbor<in_install>`; get started quickly with our :ref:`tutorials<tutorial>`; or continue reading to learn more about Arbor.

What is Arbor?
--------------

`Arbor <https://arbor-sim.org>`_ is a high-performance library for computational neuroscience simulations with multi-compartment, morphologically-detailed cells,
from single cell models to very large networks. Arbor is written from the ground up with many-cpu and gpu architectures in mind, to
help neuroscientists effectively use contemporary and future HPC systems to meet their simulation needs.

Arbor supports NVIDIA and AMD GPUs as well as explicit vectorization on CPUs from Intel (AVX, AVX2 and AVX512) and ARM (Neon and SVE).
When coupled with low memory overheads, this makes Arbor an order of magnitude faster than the most widely-used comparable simulation software.

Arbor is open source and openly developed, and we use development practices such as unit testing, continuous integration, and validation.

Documentation organisation
--------------------------

* :ref:`features-overview` is an index of Arbor's major neuroscientific features and a collection of best practices and experiences collected from the Arbor modelling community, meant to spread information on how to solve common modelling questions in Arbor.
* :ref:`tutorial` contains a few ready-made examples you can use to quickly get started using Arbor. In the tutorial descriptions we link to the relevant Arbor concepts.
* :ref:`modelintro` describes the design and concepts used in Arbor. The breakdown of concepts is mirrored (as much as possible) in the :ref:`pyoverview` and :ref:`cppoverview`, so you can easily switch between languages and concepts.
* The API section details our :ref:`pyoverview` and :ref:`cppoverview` API. The :ref:`dev-overview` describes Arbor code that is not user-facing; convenience classes, architecture abstractions, and other information that is relevant to understanding the inner workings of Arbor and the mathematical foundations underpinning the engine.
* Contributions to Arbor are very welcome! Under :ref:`contribindex` you'll find the conventions and procedures for all kinds of contributions.

Citing Arbor
------------

The Arbor software can be cited by version via Zenodo or via Arbors introductory paper.

Latest version
    |zlatest|

Version 0.10.0
    |z0100|

Version 0.9.0
    |z090|

Version 0.8.1
    |z081|

Version 0.8
    |z08|

Version 0.7
    |z07|

Version 0.6
    |z06|

Version 0.5.2
    |z052|

Version 0.2
    |z02|

Version 0.1
    |z01|

Introductory paper
    |intropaper|

    A preprint is available at `arXiv <https://arxiv.org/abs/1901.07454>`_.

Cite (Bibtex format)
    Introductory paper and latest version on Zenodo:

    .. literalinclude:: ../CITATION.bib
        :language: latex

.. |intropaper| image:: https://zenodo.org/badge/DOI/10.1109/EMPDP.2019.8671560.svg
    :target: https://doi.org/10.1109/EMPDP.2019.8671560

.. |zlatest| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13284789.svg
    :target: https://doi.org/10.5281/zenodo.13284789image::

.. |z0100| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13284789.svg
    :target: https://doi.org/10.5281/zenodo.13284789

.. |z090| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8233847.svg
    :target: https://doi.org/10.5281/zenodo.8233847
    
.. |z081| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7473671.svg
    :target: https://doi.org/10.5281/zenodo.7473671
    
.. |z08| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7323982.svg
    :target: https://doi.org/10.5281/zenodo.7323982
    
.. |z07| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6865725.svg
    :target: https://doi.org/10.5281/zenodo.6865725
    
.. |z06| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5910151.svg
    :target: https://doi.org/10.5281/zenodo.5910151

.. |z052| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5031633.svg
    :target: https://doi.org/10.5281/zenodo.5031633

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
   features/index
   tutorial/index
   ecosystem/index
   contrib/index

.. toctree::
   :caption: Concepts:
   :maxdepth: 1

   concepts/index
   concepts/recipe
   concepts/cell
   concepts/units
   concepts/interconnectivity
   concepts/hardware
   concepts/domdec
   concepts/simulation
   concepts/probe_sample

.. toctree::
   :caption: Cell Types:
   :maxdepth: 1

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
   fileformat/serdes

.. toctree::
   :caption: API reference:
   :maxdepth: 1

   python/index
   cpp/index
   dev/index


.. meta::
   :google-site-verification: KbkW8d9MLsBFZz8Ry0tfcQRkHsgxzkECCahcyRSjWDo
