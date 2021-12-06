import os
import sys
from skbuild import setup

# VERSION is in the same path as setup.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'VERSION')) as version_file:
    version_ = version_file.read().strip()

# Get the contents of the readme
with open(os.path.join(here, 'python/readme.md'), encoding='utf-8') as f:
    long_description = f.read()

opt = {'mpi': False,
                               'gpu': 'none',
                               'vec': False,
                               'arch': 'none',
                               'neuroml': True,
                               'bundled': True,
                               'makejobs': 2}
setup(
    name='arbor',
    version=version_,
    python_requires='>=3.6',

    install_requires=['numpy'],
    setup_requires=[],
    zip_safe=False,
    cmake_args = [
        '-DARB_WITH_PYTHON=on',
        '-DPYTHON_EXECUTABLE=' + sys.executable,
        '-DARB_WITH_MPI={}'.format( 'on' if opt['mpi'] else 'off'),
        '-DARB_VECTORIZE={}'.format('on' if opt['vec'] else 'off'),
        '-DARB_ARCH={}'.format(opt['arch']),
        '-DARB_GPU={}'.format(opt['gpu']),
        '-DARB_WITH_NEUROML={}'.format( 'on' if opt['neuroml'] else 'off'),
        '-DARB_USE_BUNDLED_LIBS={}'.format('on' if opt['bundled'] else 'off'),
        '-DCMAKE_BUILD_TYPE=Release' # we compile with debug symbols in release mode.
    ],
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
        'Programming Language :: C++',
    ],
    project_urls={
        'Source': 'https://github.com/arbor-sim/arbor',
        'Documentation': 'https://docs.arbor-sim.org',
        'Bug Reports': 'https://github.com/arbor-sim/arbor/issues',
    },
)
