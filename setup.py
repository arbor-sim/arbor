import os
import sys
from skbuild import setup

# Hard coded, because scikit-build does not do build options.
# Override by instructing CMAKE, e.g.:
# pip install . -- -DARB_USE_BUNDLED_LIBS=ON -DARB_WITH_MPI=ON -DARB_GPU=cuda
opt = {'mpi': False,
       'gpu': 'none',
       'vec': False,
       'arch': 'none',
       'neuroml': True,
       'bundled': True}

# VERSION is in the same path as setup.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'VERSION')) as version_file:
    version_ = version_file.read().strip()

# Get the contents of the readme
with open(os.path.join(here, 'python/readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='arbor',
      version=version_,
      python_requires='>=3.6',
      install_requires=['numpy'],
      setup_requires=[],
      zip_safe=False,
      cmake_args = ['-DARB_WITH_PYTHON=on',
                    '-DPYTHON_EXECUTABLE=' + sys.executable,
                    f'-DARB_WITH_MPI={opt["mpi"]}',
                    f'-DARB_VECTORIZE={opt["vec"]}'
                    f'-DARB_ARCH={opt["arch"]}',
                    f'-DARB_GPU={opt["gpu"]}',
                    f'-DARB_WITH_NEUROML={opt["neuroml"]}',
                    f'-DARB_USE_BUNDLED_LIBS={opt["bundled"]}',
                    '-DCMAKE_BUILD_TYPE=Release'],
      author='The Arbor dev team.',
      url='https://github.com/arbor-sim/arbor',
      description='High performance simulation of networks of multicompartment neurons.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: C++',
      ],
      project_urls={'Source': 'https://github.com/arbor-sim/arbor',
                    'Documentation': 'https://docs.arbor-sim.org',
                    'Bug Reports': 'https://github.com/arbor-sim/arbor/issues',},)
