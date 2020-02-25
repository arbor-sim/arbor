import os
import sys
import setuptools
import pathlib
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import subprocess

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'VERSION')) as version_file:
    version_ = version_file.read().strip()

class cmake_extension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class cmake_build(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        build_directory = os.path.abspath(self.build_temp)

        cmake_args = [
            '-DARB_WITH_PYTHON=on',
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Assuming Makefiles
        build_args += ['--', '-j4']

        self.build_args = build_args

        env = os.environ.copy()
        env['CXXFLAGS'] = '{}'.format(env.get('CXXFLAGS', ''))
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print('-'*10, 'Running CMake prepare', '-'*40)
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                              cwd=self.build_temp, env=env)

        print('-'*10, 'Building extensions', '-'*40)
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        print('CMake command: ', cmake_cmd)
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = pathlib.Path(self.build_temp).resolve()
        dest_path = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / self.get_ext_filename(ext.name)
        print('build_temp: ', build_temp)
        print('dest_path: ', dest_path)
        print('source_path: ', source_path)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)

setuptools.setup(
    name='arbor',
    #packages=['arbor'],
    version=version_,
    #package_data={
        #'arbor': ['VERSION', '_arbor.*.so'],
    #},
    python_requires='>=3.6',
    install_requires=[],
    setup_requires=[],
    zip_safe=False,
    ext_modules=[cmake_extension('arbor')],
    cmdclass=dict(build_ext=cmake_build),

    author='CSCS and FSJ',
    url='https://github.com/arbor-sim/arbor',
    description='High performance simulation of networks of multicompartment neurons.',
    long_description='',
    classifiers=[
        'Development Status :: 4 - Beta', # Upgrade to "5 - Production/Stable" on release.
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Build Tools',
        'License :: OSI Approved :: BSD License'
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

