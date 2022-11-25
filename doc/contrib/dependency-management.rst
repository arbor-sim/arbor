.. _contribdepverman:

Dependency management
=====================

Arbor relies on a (small) number of dependencies. We can distinguish three kinds of deps:

0. Source management dependencies: Git.
1. Build dependencies. E.g. CMake, compilers like GCC or CUDA.
2. Linking dependencies. E.g. MPI.
3. Source dependencies. These are present as `git submodules <https://git-scm.com/docs/git-submodule>`_ or as copy in ``ext/``. Their use is optional: users who need integration with their package manager (e.g. apt, spack, yum) can link to those versions instead.

Note that the actual dependencies of your build configuration may vary.

In addition, ``spack/package.py`` contains a copy of the Spack package definition `upstream <https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/arbor/package.py>`_. Here instructions for both in-repo and configure-time dependencies are defined.

This document contains rules for when and how to update dependencies, and what to be mindful of when you do.

List of dependencies
--------------------

A full list of dependencies is maintained at ``doc/dependencies.csv``:

.. csv-table:: List of configure time dependencies
   :file: ../dependencies.csv
   :widths: 10, 20, 10, 70, 1
   :header-rows: 1

.. note::

   CMake can generate an overview of the dependency tree. Run the following in the build dir:

   .. code-block:: bash

      cmake --graphviz=graphviz/arbor-dep-graph.dot . && dot graphviz/arbor-dep-graph.dot -T png -o graphviz/arbor-dep-graph.png

   This plot can be tweaked with the ``CMakeGraphVizOptions.cmake`` file in the root of the project, which currently excludes tests, the ``lmorpho`` morphology generator and all C++ examples.

User platforms
--------------

Although Arbor should in principle run everywhere modern compilers are found, a couple of user platforms
are essential. These environments must be able to build Arbor without issue, if not we consider this a bug.
Also, build instructions for each of them must be given in the documentation.

* Ubuntu LTS-latest
* Ubuntu LTS-latest-1
* MacOS-latest
* MacOS-latest-1
* Cray programming environment on Piz Daint
* Programming environment on Juwels Booster (**todo** CI at JSC)
* Github Action venvs, see `list <https://github.com/actions/virtual-environments>`_.
* Manylinux containers. For compatibility of manylinux tags, see `here <https://github.com/pypa/manylinux#readme>`_.

Dependency update rules
-----------------------

#. ``doc/dependencies.csv``, git submodules and ``spack/package.py`` shall be in sync.
#. Dependencies shall be set to a (commit hash corresponding to a) specific version tag. (All current dependencies use semver.)
#. The version shall be compatible with the user platforms (see above).
#. The version shall be compatible with the requirements in ``doc/dependencies.csv``.
#. The version shall be the lowest possible. More recent versions of submodules are automatically tested through ``.github/workflows/check-submodules.yml``, to catch merge problems early.
#. Moreover, dependencies shall not be updated past the most recent version of the dependency in Spack.

   * This prevents Spack builds from pulling in ``master``, when a more recent version than available is required. `See here <https://spack.readthedocs.io/en/latest/packaging_guide.html#version-comparison>`_.
   * This is a manual check, e.g. for pybind: `see pybind package.py <https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/py-pybind11/package.py>`_
#. Actually updating shall remain a manual process. Update may require nontrivial updates to Arbor, and updates to Spack upstream (e.g. make PR for pybind update).
#. A dependency update shall have a separate PR, and such a PR updates a single dependency at a time, unless the dependency update requires other dependencies to be updated.
#. This PR requires review by at least two reviewers.

   * There appears to be no way to enforce that, unless we enforce it for all PRs.
   * Optionally we could have a PR template auto-assigning to some or all of us, which means we'll at least be notified.
#. We will try to keep compatible to a wide range in dependency versions.

   * This includes making other components in `ext` git submodules, such that updates are more easily tracked.
