Releases
********

Release cycle
=============

0. release every 3 months (at time ``T``)
1. ``T-11`` weeks: ``all`` add your favorite Issues to the next-rel column
2. ``T-10`` weeks: ``Scrum Master`` prep dev meet (internal)

   * Update/trim next-release column in Kanban
   * Prepare agenda, include possible additions not covered by Kanban/Issues
   * Add milestone tags (nextver, nextver+1, etc.)
3. ``T-8`` weeks: ``Release Manager`` dev meet (external/public)

   * Use Kanban as starter
   * Move issues around based on input
   * Add milestone tags, for this release or future releases
4. ``T±0``: ``Release Manager`` release!
5. ``T+1`` weeks: ``Scrum Master`` retrospective
   
   * set date for next release

Procedure
=========

These notes enumerate the steps required every time we release a new
version of Arbor.

Pre-release
-----------

Update tags/versions and test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0. Check README.md, ATTRIBUTIONS.md, CONTRIBUTING.md.
1. Create new temp-branch ending in ``-rc``. E.g. ``v0.6-rc``
2. Bump the ``VERSION`` file:
   https://github.com/arbor-sim/arbor/blob/master/VERSION
3. Run all tests.
   - ``ciwheel.yml`` does this, and also runs a build from sdist, which triggers weekly and on new (RC) tags. Make sure this is OK.
   - This should catch many problems. For a manual check:
   - Verify MANIFEST.in (required for PyPI sdist)
   - Check Python/pip/PyPi metadata and scripts, e.g. ``setup.py``
   - Double check that all examples/tutorials/etc are covered by CI

Test the RC
~~~~~~~~~~~

4. Collect artifact from the above GA run.
   In case you want to manually want to trigger ``ciwheel.yml`` GA, overwrite the ``ciwheel`` branch with the commit of your choosing and force push to Github.
5. twine upload -r testpypi dist/\*
6. Ask users to test the above, e.g.:

.. code-block:: bash

   python -m venv env && source env/bin/activate
   pip install numpy
   pip install -i https://test.pypi.org/simple/ arbor==0.6-rc
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
