import os
import sys
from skbuild import setup
import subprocess

# Singleton class that holds the settings configured using command line
# options. This information has to be stored in a singleton so that it
# can be passed between different stages of the build, and because pip
# has strange behavior between different versions.
class CL_opt:
    instance = None
    def __init__(self):
        if not CL_opt.instance:
            CL_opt.instance = {'mpi': False,
                               'gpu': 'none',
                               'vec': False,
                               'arch': 'none',
                               'neuroml': True,
                               'bundled': True,
                               'makejobs': 2}

    def settings(self):
        return CL_opt.instance

def cl_opt():
    return CL_opt().settings()

# extend user_options the same way for all Command()s
user_options_ = [
        ('mpi',   None, 'enable mpi support (requires MPI library)'),
        ('gpu=',  None, 'enable nvidia cuda support (requires cudaruntime and nvcc) or amd hip support. Supported values: '
                        'none, cuda, cuda-clang, hip'),
        ('vec',   None, 'enable vectorization'),
        ('arch=', None, 'cpu architecture, e.g. haswell, skylake, armv8.2-a+sve, znver2 (default native).'),
        ('neuroml', None, 'enable parsing neuroml morphologies in Arbor (requires libxml)'),
        ('sysdeps', None, 'don\'t use bundled 3rd party C++ dependencies (pybind11 and json). This flag forces use of dependencies installed on the system.'),
        ('makejobs=', None, 'the amount of jobs to run `make` with.')
    ]

# VERSION is in the same path as setup.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'VERSION')) as version_file:
    version_ = version_file.read().strip()

# Get the contents of the readme
with open(os.path.join(here, 'python/readme.md'), encoding='utf-8') as f:
    long_description = f.read()

class _command_template:
    """
    Override a setuptools-like command to augment the command line options.
    Needs to appear before the command class in the class's argument list for
    correct MRO.

    Examples
    --------

    .. code-block: python

      class install_command(_command_template, install):
          pass


      class complex_command(_command_template, mixin1, install):
          def initialize_options(self):
              # Both here and in `mixin1`, a `super` call is required
              super().initialize_options()
              # ...
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.user_options = super().user_options + user_options_


    def initialize_options(self):
        super().initialize_options()
        self.mpi  = None
        self.gpu  = None
        self.arch = None
        self.vec  = None
        self.neuroml = None
        self.sysdeps = None
        self.makejobs = 2

    def finalize_options(self):
        super().finalize_options()
        try:
            self.makejobs = int(self.makejobs)
        except ValueError:
            err = True
        else:
            err = False
        if err or self.makejobs < 1:
            raise AssertionError('makejobs must be a strictly positive integer')

    def run(self):
        # The options are stored in global variables:
        opt = cl_opt()
        #   mpi  : build with MPI support (boolean).
        opt['mpi']  = self.mpi is not None
        #   gpu  : compile for AMD/NVIDIA GPUs and choose compiler (string).
        opt['gpu']  = "none" if self.gpu is None else self.gpu
        #   vec  : generate SIMD vectorized kernels for CPU micro-architecture (boolean).
        opt['vec']  = self.vec is not None
        #   arch : target CPU micro-architecture (string).
        opt['arch'] = 'none' if self.arch is None else self.arch
        #   neuroml : compile with neuroml support for morphologies.
        opt['neuroml'] = self.neuroml is not None
        #   bundled : use bundled/git-submoduled 3rd party libraries.
        #             By default use bundled libs.
        opt['bundled'] = self.sysdeps is None
        #   makejobs : specify amount of jobs.
        #              By default 2.
        opt['makejobs'] = int(self.makejobs)

        super().run()

opt = cl_opt()
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
