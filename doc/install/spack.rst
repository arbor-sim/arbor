.. _in_spack:

Spack Installation
===================

Install Arbor
-------------

To install Arbor using Spack, run ``spack install arbor``.

Build Options
-------------

Arbor can be built with various options, just like the regular CMake build. For instance, to have Spack build Arbor with MPI enabled, run ``spack install arbor +mpi``. For a full overview of the build options, please refer to the `our Spack package.yml <https://github.com/arbor-sim/arbor/blob/master/spack/package.py>`_.

Why use Spack?
--------------

`Spack <https://spack.io>`_ is a package manager for supercomputers, Linux, and macOS. It makes installing scientific software easy. Spack isn’t tied to a particular language; you can build a software stack in Python or R, link to libraries written in C, C++, or Fortran, and easily swap compilers or target specific microarchitectures.

A powerful feature for users of scientific software is Spack's `Environment feature <https://spack.readthedocs.io/en/latest/environments.html>`_. One can define and store software environments for reuse, to generate container images or reproduce and rerun software workflows at a later time.

Issues when using Spack
-----------------------

On some systems initial Spack setup requires an extra step currently not shown in the up-front installations instructions of the `Spack documentation <https://spack.readthedocs.io>`_, which is adding the compilers on your system to Spack's configuration. If you don't, you may get this error:

.. code-block:: bash

    No satisfying compiler available is compatible with a satisfying os

The solution is to run (`as described further down in the official documentation <https://spack.readthedocs.io/en/latest/getting_started.html#compiler-configuration>`_):

.. code-block:: bash

    ./spack compiler add

To get help in case of problems, please make an issue at `Arbor's issues page <https://github.com/arbor-sim/arbor/issues>`_.
