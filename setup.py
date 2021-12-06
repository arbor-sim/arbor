from pathlib import Path
import sys
from skbuild import setup
from argparse import ArgumentParser

P = ArgumentParser(description='Arbor build options.')
P.add_argument('--vec',     dest='vec',        action='store_const', const='on', default='off',  help='Enable SIMD.')
P.add_argument('--nml',     dest='nml',        action='store_const', const='on', default='off',  help='Enable NeuroML2 support. Requires libxml2.')
P.add_argument('--mpi',     dest='mpi',        action='store_const', const='on', default='off',  help='Enable MPI.')
P.add_argument('--bundled', metavar='bundled', action='store_const', const='on', default='off',  help='Use bundled libs.')
P.add_argument('--gpu',     metavar='gpu',                                       default='none', help='Enable GPU support.')
P.add_argument('--arch',    metavar='arch',                                      default='',     help='Set processor architecture.')
opt, _ = P.parse_known_args()

# Get directory in which this script resides
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
      cmake_args = ['-DARB_WITH_PYTHON=on',
                    '-DCMAKE_BUILD_TYPE=Release'
                    f'-DPYTHON_EXECUTABLE={sys.executable}',
                    f'-DARB_WITH_MPI={opt.mpi}',
                    f'-DARB_VECTORIZE={opt.vec}'
                    f'-DARB_ARCH={opt.arch}',
                    f'-DARB_GPU={opt.gpu}',
                    f'-DARB_WITH_NEUROML={opt.nml}',
                    f'-DARB_USE_BUNDLED_LIBS={opt.bundled}',],
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
                   'Programming Language :: C++',],
      project_urls={'Source': 'https://github.com/arbor-sim/arbor',
                    'Documentation': 'https://docs.arbor-sim.org',
                    'Bug Reports': 'https://github.com/arbor-sim/arbor/issues',},)
