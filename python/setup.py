import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'arbor/VERSION')) as version_file:
    version_ = version_file.read().strip()

setuptools.setup(
    name='arbor',
    packages=['arbor'],
    version=version_,
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
    package_data={
        'arbor': ['VERSION', '_arbor.*.so'],
    },
    python_requires='>=3.6',
    install_requires=[],
    setup_requires=[],
    zip_safe=False,
)

