Releases
********

Release cycle
=============

0. release every 3 months (at time ``T``)
1. ``T-11`` weeks: ``all`` Add your favorite Issues to the next-rel column
2. ``T-10`` weeks: ``AK`` prep dev meet (internal)

   * Update/trim next-release column in Kanban.
   * Add milestone tags (nextver, nextver+1, etc.)
3. ``T-8`` weeks: ``BH`` dev meet (external/public)

   * Use Kanban as starter.
   * Move issues around based on input.
   * Add milestone tags, for this release or future releases
4. ``T±0``: ``BH`` release!
5. ``T+1`` weeks: ``AK`` retrospective two weeks after release

Procedure
=========

These notes enumerate the steps required every time we release a new
version of Arbor.

Pre-release
-----------

Update tags/versions and test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create new temp-branch ending in ``-rc``. E.g. ``v0.5.1-rc``
2. Bump the ``VERSION`` file:
   https://github.com/arbor-sim/arbor/blob/master/VERSION
3. Update Python/pip/PyPi metadata and scripts

   - Update MANIFEST (required for PyPi step later):
     https://github.com/arbor-sim/arbor/blob/master/MANIFEST.in
   - also checkout ``setup.cfg`` and ``setup.py``

4. Double check all examples/tutorials/etc not covered by CI

Manual test (deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~

5.  python setup.py sdist
6.  python -m venv env
7.  source env/bin/activate
8.  move tarball here and extract
9.  pip install –upgrade pip
10. pip install numpy
11. pip install ./arbor-0.5.1 –verbose
12. python -c ’import arbor; print(arbor.__config__)’
13. twine upload -r testpypi dist/\* (have some testrepo)
14. create *another* venv: python -m venv env && source env/bin/activate
15. pip install numpy
16. pip install -i https://test.pypi.org/simple/ arbor==0.5.1 –verbose
17. python -c ’import arbor; print(arbor.__config__)’

Ciwheel/automated test (replaces manual test)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5. Create/overwrite ``ciwheel`` branch with the above branch, and push
   to Github.
6. Collect artifact from GA run.
7. twine upload -r testpypi dist/\*
8. Ask users to test the above, e.g.:

.. code-block:: bash

   python -m venv env && source env/bin/activate
   pip install numpy pip install -i https://test.pypi.org/simple/ arbor==0.5.1
   python -c ’import arbor; print(arbor.__config__)’

Release
-------

0. Make sure ciwheel passes tests, produced working wheels. Make sure
   tests on master also passed, and master == ciwheel
1. Tag and release: https://github.com/arbor-sim/arbor/releases

   -  on cmdline: git tag -a TAGNAME
   -  git push origin TAGNAME
   -  Go to `GH tags`_ and click “…” and “Create release”
   -  Go through merged PRs to come up with a changelog

2. Create tarball with
   ``scripts/create_tarball ~/loc/of/arbor tagname outputfile``

   -  eg ``scripts/create_tarball /full/path/to/arbor v0.5.1 ~/arbor-v0.5.1-full.tar.gz``

3. [`AUTOMATED`_] push to git@gitlab.ebrains.eu:arbor-sim/arbor.git
4. Download output of wheel action and extract (verify the wheels and
   source targz is in /dist)
5. Verify wheel

   -  create venv: python -m venv env && source env/bin/activate
   -  pip install arbor-0.5.1-cp39-cp39-manylinux2014_x86_64.whl
   -  python -c ’import arbor; print(arbor.__config__)’

6. Upload to pypi

   -  twine upload -r arborpypi dist/\*

7. Verify

   -  create venv: python -m venv env && source env/bin/activate
   -  pip install arbor==0.5.1 –verbose
   -  python -c ’import arbor; print(arbor.__config__)’

Post release
------------

1. Update spack package

   -  first, update ``spack/package.py``. The checksum of the targz is the sha256sum.
   -  Then, use the file to `make PR here <https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/>`_

.. _GH tags: https://github.com/arbor-sim/arbor/tags
.. _AUTOMATED: https://github.com/arbor-sim/arbor/blob/master/.github/workflows/ebrains.yml 
