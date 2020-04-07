import os
import sys
import setuptools
import pathlib
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import subprocess

# VERSION is in the same path as setup.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'VERSION')) as version_file:
    version_ = version_file.read().strip()

# Get the contents of the readme
with open(os.path.join(here, 'python/readme.md'), encoding='utf-8') as f:
    long_description = f.read()

def check_cmake():
    try:
        out = subprocess.check_output(['cmake', '--version'])
        return True
    except OSError:
        return False

# Extend the command line options available to the install phase.
# These arguments must come after `install` on the command line, e.g.:
#    python3 setup.py install --mpi --arch=skylake
#    pip3 install --install-option '--mpi' --install-option '--arch=skylake' .
class install_command(install):
    user_options = install.user_options + [
        ('mpi',   None, 'enable mpi support (requires MPI library)'),
        ('gpu',   None, 'enable nvidia cuda support (requires cudaruntime and nvcc)'),
        ('vec',   None, 'enable vectorization'),
        ('arch=', None, 'cpu architecture, e.g. haswell, skylake, armv8-a'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.mpi  = None
        self.gpu  = None
        self.arch = None
        self.vec  = None

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # The options are stored in global variables:
        #   cl_opt_mpi  : build with MPI support (boolean).
        #   cl_opt_gpu  : build with CUDA support (boolean).
        #   cl_opt_vec  : generate SIMD vectorized kernels for CPU micro-architecture (boolean).
        #   cl_opt_arch : target CPU micro-architecture (string).
        global cl_opt_mpi
        global cl_opt_gpu
        global cl_opt_vec
        global cl_opt_arch
        cl_opt_mpi  = self.mpi is not None
        cl_opt_gpu  = self.gpu is not None
        cl_opt_vec  = self.vec is not None
        cl_opt_arch = "native" if self.arch is None else self.arch

        install.run(self)

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
            '-DARB_WITH_MPI={}'.format( 'on' if cl_opt_mpi else 'off'),
            '-DARB_WITH_GPU={}'.format( 'on' if cl_opt_gpu else 'off'),
            '-DARB_VECTORIZE={}'.format('on' if cl_opt_vec else 'off'),
            '-DARB_ARCH={}'.format(cl_opt_arch),
            '-DCMAKE_BUILD_TYPE=Release' # we compile with debug symbols in release mode.
        ]

        print('-'*5, 'cmake arguments: {}'.format(cmake_args))

        build_args = ['--config', 'Release']

        # Assuming Makefiles
        build_args += ['--', '-j2']

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

    install_requires=[],
    setup_requires=[],
    zip_safe=False,
    ext_modules=[cmake_extension('arbor')],
    cmdclass={
        'build_ext': cmake_build,
        'install':   install_command,
    },

    author='The lovely Arbor devs.',
    url='https://github.com/arbor-sim/arbor',
    description='High performance simulation of networks of multicompartment neurons.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta', # Upgrade to "5 - Production/Stable" on release of v0.3
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    project_urls={
        'Source': 'https://github.com/arbor-sim/arbor',
        'Documentation': 'https://arbor.readthedocs.io',
        'Bug Reports': 'https://github.com/arbor-sim/arbor/issues',
    },
)

