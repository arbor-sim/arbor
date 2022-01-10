from pathlib import Path
from sys import executable as python
from skbuild import setup

# Hard coded options, because scikit-build does not do build options.
# Override by instructing CMAKE, e.g.:
# pip install . -- -DARB_USE_BUNDLED_LIBS=ON -DARB_WITH_MPI=ON -DARB_GPU=cuda
with_mpi   = False
with_gpu   = 'none'
with_vec   = False
arch       = 'none'
with_nml   = True
use_libs   = True
build_type = 'Release' # this is ok even for debugging, as we always produce info

# Find our dir; *should* be the arbor checkout
here = Path(__file__).resolve().parent
# Read version file
with open(here / 'VERSION') as fd:
    arbor_version = fd.read().strip()
# Get the contents of the readme
with open(here / 'python' / 'readme.md', encoding='utf-8') as fd:
    long_description = fd.read()

setup(name='arbor',
      version=arbor_version,
      python_requires='>=3.6',
      install_requires=['numpy'],
      setup_requires=[],
      zip_safe=False,
      cmake_args=['-DARB_WITH_PYTHON=on',
                  f'-DPYTHON_EXECUTABLE={python}',
                  f'-DARB_WITH_MPI={with_mpi}',
                  f'-DARB_VECTORIZE={with_vec}'
                  f'-DARB_ARCH={arch}',
                  f'-DARB_GPU={with_gpu}',
                  f'-DARB_WITH_NEUROML={with_nml}',
                  f'-DARB_USE_BUNDLED_LIBS={use_libs}',
                  f'-DCMAKE_BUILD_TYPE={build_type}',],
      author='The Arbor dev team.',
      url='https://github.com/arbor-sim/arbor',
      description='High performance simulation of networks of multicompartment neurons.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   'Programming Language :: C++',],
      project_urls={'Source': 'https://github.com/arbor-sim/arbor',
                    'Documentation': 'https://docs.arbor-sim.org',
                    'Bug Reports': 'https://github.com/arbor-sim/arbor/issues',},)
