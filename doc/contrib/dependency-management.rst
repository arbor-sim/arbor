.. _contribdepverman:

Dependency management
=====================

Arbor relies on a (small) number of dependencies. Some are configured before building (e.g. a C++ compiler, optionally MPI bindings, libxml2, etc.), and some are present in the repo, either as a `git submodule <https://git-scm.com/docs/git-submodule>`_ or as copy.

The in-repo dependencies are located in ``ext/``, with the exception of `Pybind11 <https://github.com/pybind/pybind11>`_, which lives in ``python/pybind11``.

In addition, ``spack/package.py`` contains a copy of the Spack package definition `upstream <https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/arbor/package.py>`_. Here instructions for both in-repo and configure-time dependencies are defined.

This document contains rules for when and how to update dependencies, and what to be mindful of when you do.

List of dependencies
--------------------

In repo deps: see earlier comment.

The rest: ``doc/dependencies.csv``:

.. csv-table:: List of configure time dependencies
   :file: ../dependencies.csv
   :widths: 10, 20, 10, 70, 1
   :header-rows: 1

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

1. Submodule update notification occurs through ``.github/workflows/check-submodules.yml``.
2. Dependencies shall be set to a (commit hash corresponding to a) specific version tag. (All current dependencies use semver.)
3. Git submodules and dependencies in ``spack/package.py`` shall be in sync.
4. The version shall be compatible with the user platforms (see above).
5. The version shall be compatible with the requirements in ``doc/dependencies.csv``.
6. Submodules shall not be updated past the most recent version of the dependency in Spack.

   * This prevents Spack builds from pulling in ``master``, when a more recent version than available is required. `See here <https://spack.readthedocs.io/en/latest/packaging_guide.html#version-comparison>`_.
   * This is a manual check, e.g. for pybind: `see pybind package.py <https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/py-pybind11/package.py>`_
7. Actually updating shall remain a manual process. Update may require nontrivial updates to Arbor, and updates to Spack upstream (e.g. make PR for pybind update).
8. A dependency update shall have a separate PR, and such a PR updates a single dependency at a time, unless the dependency update requires other dependencies to be updated.
9. This PR requires review by at least two reviewers.

   * There appears to be no way to enforce that, unless we enforce it for all PRs.
   * Optionally we could have a PR template auto-assigning to some or all of us, which means we'll at least be notified.
10. We will try to keep deps up to date.

   * This includes making other components in `ext` git submodules, such that updates are more easily tracked.