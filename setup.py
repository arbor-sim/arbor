import os
import sys
import setuptools
import pathlib
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import subprocess
try:
    from wheel.bdist_wheel import bdist_wheel
    WHEEL_INSTALLED = True
except:
    #wheel package not installed.
    WHEEL_INSTALLED = False
    pass

opt = {} #global placeholder for install options. Wheels can not depend on any flags passed/options set, so the defaults in this file must match the instructions for building a Wheel

# VERSION is in the same path as setup.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'VERSION')) as version_file:
    version_ = version_file.read().strip()

# Get the contents of the readme
with open(os.path.join(here, 'python/readme.md'), encoding='utf-8') as f:
    long_description = f.read()

# extend user_options the same way for all Command()s
user_options_ = [
        ('mpi',   None, 'enable mpi support (requires MPI library)'),
        ('gpu=',  None, 'enable nvidia cuda support (requires cudaruntime and nvcc) or amd hip support. Supported values: '
                        'none, cuda, cuda-clang, hip'),
        ('vec',   None, 'enable vectorization'),
        ('arch=', None, 'cpu architecture, e.g. haswell, skylake, armv8.2-a+sve, znver2 (default native).'),
        ('noneuroml', None, 'disable neuroml support (set if you don\'t have libxml)'),
        ('nobundled', None, 'don\'t use bundled 3rd party C++ dependencies (pybind11 and json). This flag forces use of dependencies installed on the system.'),
        ('makejobs=', None, 'the amount of jobs to run `make` with.')
    ]

def check_cmake():
    try:
        out = subprocess.check_output(['cmake', '--version'])
        return True
    except OSError:
        return False


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
        self.noneuroml = None
        self.nobundled = None
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
        global opt
        #   mpi  : build with MPI support (boolean).
        opt['mpi']  = self.mpi is not None
        #   gpu  : compile for AMD/NVIDIA GPUs and choose compiler (string).
        opt['gpu']  = "none" if self.gpu is None else self.gpu
        #   vec  : generate SIMD vectorized kernels for CPU micro-architecture (boolean).
        opt['vec']  = self.vec is not None
        #   arch : target CPU micro-architecture (string).
        opt['arch'] = 'none' if self.arch is None else self.arch
        #   neuroml : compile with neuroml support for morphologies.
        opt['neuroml'] = self.noneuroml is None
        #   bundled : use bundled/git-submoduled 3rd party libraries.
        #             By default use bundled libs.
        opt['bundled'] = self.nobundled is None
        #   makejobs : specify amount of jobs.
        #              By default 2.
        opt['makejobs'] = int(self.makejobs)

        print(self)
        print(opt)

        super().run()


class install_command(_command_template, install):
    pass

if WHEEL_INSTALLED:
    class bdist_wheel_command(_command_template, bdist_wheel):
        pass


class cmake_extension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class cmake_build(build_ext):
    def run(self):
        if not check_cmake():
            raise RuntimeError('CMake is not available. CMake 3.12 is required.')

        # The path where CMake will be configured and Arbor will be built.
        build_directory = os.path.abspath(self.build_temp)
        # The path where the package will be copied after building.
        lib_directory = os.path.abspath(self.build_lib)
        # The path where the Python package will be compiled.
        source_path = build_directory + '/python/arbor'
        # Where to copy the package after it is built, so that whatever the next phase is
        # can copy it into the target 'prefix' path.
        dest_path = lib_directory + '/arbor'

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
        ]

        print('-'*5, 'command line arguments: {}'.format(opt))
        print('-'*5, 'cmake arguments: {}'.format(cmake_args))

        build_args = ['--config', 'Release']

        # Assuming Makefiles
        build_args += ['--', f'-j{opt["makejobs"]}']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{}'.format(env.get('CXXFLAGS', ''))
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print('-'*20, 'Configure CMake')
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                              cwd=self.build_temp, env=env)

        print('-'*20, 'Build')
        cmake_cmd = ['cmake', '--build', '.'] + build_args
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        # Copy from build path to some other place from whence it will later be installed.
        # ... or something like that
        # ... setuptools is an enigma monkey patched on a mystery
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)
        self.copy_tree(source_path, dest_path)

setuptools.setup(
    name='arbor',
    version=version_,
    python_requires='>=3.6',
    install_requires=['numpy'],
    setup_requires=[],
    zip_safe=False,
    ext_modules=[cmake_extension('arbor')],
    cmdclass={
        'build_ext':   cmake_build,
        'install':     install_command,
        'bdist_wheel': bdist_wheel_command,
    } if WHEEL_INSTALLED else {
        'build_ext':   cmake_build,
        'install':     install_command,
    },

    author='The Arbor dev team.',
    url='https://arbor-sim.org',
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
    project_urls={
        'Source': 'https://github.com/arbor-sim/arbor',
        'Documentation': 'https://docs.arbor-sim.org',
        'Bug Reports': 'https://github.com/arbor-sim/arbor/issues',
    },
)
