.. _contribrelease:

Releasing Arbor
===============

These notes enumerate the steps required every time we release a new
version of Arbor.

Before release
~~~~~~~~~~~~~~

1. Bump the ``VERSION`` file:
   https://github.com/arbor-sim/arbor/blob/master/VERSION
2. Update Python/pip/PyPi metadata and scripts

-  Update MANIFEST (required for PyPi step later):
   https://github.com/arbor-sim/arbor/blob/master/MANIFEST.in
-  also checkout ``setup.cfg`` and ``setup.py``

3. Double check all examples/tutorials/etc not covered by CI

Release
~~~~~~~

4. Tag and release: https://github.com/arbor-sim/arbor/releases

Post release
~~~~~~~~~~~~

5. Upload new version to PyPi

-  TODO put the steps here, as soon as I figure them out again

6. Update spack package
7. Add release for citation on Zenodo
8. Add tagged version of docs on ReadTheDocs
9. HBP internal admin

-  `Plus <https://plus.humanbrainproject.eu/components/2691/>`__
-  `TC
   Wiki <https://wiki.ebrains.eu/bin/view/Collabs/technical-coordination/EBRAINS%20components/Arbor/>`__
-  `KG <https://kg.ebrains.eu/search/instances/Software/80d205a9-ffb9-4afe-90b8-2f12819950ec>`__
   - (reportedly unimportant, listed for completeness sake)
-  Send an update to the folk in charge of HBP Twitter if we want to
   shout about it

Python configuration
--------------------

Here is a description of the steps required to release a version of
Arbor to PyPi so that it can be installed using pip. In the example
below, we are working with v0.4.

**Quick steps to test installation**

::

   python -m venv env
   source env/bin/activate
   pip install --upgrade pip

   # test out basic install
   pip install git+https://github.com/arbor-sim/arbor.git@v0.4-rc --verbose

   # test out a fancy install (depends on what you have available)
   pip install --install-option='--vec' --install-option='--gpu=cuda' --install-option='--mpi' git+https://github.com/arbor-sim/arbor.git@v0.4-rc --verbose

   # test out installation
   python -c 'import arbor; print(arbor.__config__)'

   # I get the following output:
   #   {'mpi': False, 'mpi4py': False, 'gpu': True, 'vectorize': True, 'version': '0.4', 'source': '2020-10-07 21:06:47 +0200 4a94032abe2925e462727400105c6c55ef4d87c5', 'arch': 'native'}

set up a clean environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python3 -m venv env
   source env/bin/activate
   pip install --upgrade pip
   pip install wheel
   pip install twine

create a new branch
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git checkout -b v0.4-rc

update VERSION
~~~~~~~~~~~~~~

.. code:: bash

   vim VERSION

test
~~~~

::

   git push origin v0.4-rc

   # try to install in your virtual env

   pip install git+https://github.com/arbor-sim/arbor.git@v0.4-rc

   # try out some different options

   pip install --install-option='--vec' --install-option='--gpu=cuda' git+https://github.com/arbor-sim/arbor.git@v0.4-rc

create source distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python setup.py sdist
   # check the distribution:
   ls dist
   # prints the following for this use case:
   #     arbor-0.4.tar.gz

The source distribution can be expanded somewhere and you can try to
install from there. This can save you the latency of uploading it to
TestPyPi to discover that there was an issue.

.. code:: bash

   cp dist/arbor-0.4.tar.gz ~/tmp
   cd ~/tmp
   tar -xzvf arbor-0.4.tar.gz
   cd arbor-0.4
   # you can check the auto-generated PKF-INFO file.
   # check for 'UNKNOWN' in PKG-INFO, which usually indicates incorrect/missing information
   # in the setup.py file.
   python setup.py install

upload it to testpypi
=====================

::

   python3 -m twine upload --repository testpypi dist/*

**note** give the package a different name in the ``config.py`` if you
have to upload multiple times.
