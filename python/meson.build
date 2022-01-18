pyarb_srcs = [
  'cable_cell_io.cpp',
  'cable_probes.cpp',
  'cells.cpp',
  'config.cpp',
  'context.cpp',
  'domain_decomposition.cpp',
  'error.cpp',
  'event_generator.cpp',
  'identifiers.cpp',
  'mechanism.cpp',
  'morphology.cpp',
  'mpi.cpp',
  'profiler.cpp',
  'pyarb.cpp',
  'recipe.cpp',
  'schedule.cpp',
  'simulation.cpp',
  'single_cell_model.cpp',
]

if arb_with_bundled_libs
  dep_pybind   = declare_dependency(include_directories: ['pybind11/include'])
else
  dep_pybind   = dependency('pybind11', version : '>=2.8.1', required : true)
endif

python3 = import('python').find_installation('python3')
dep_python = dependency('python3', required: true)

py_dependencies = [dep_python, dep_pybind, dep_arborio, dep_arb]
if arb_with_gpu
  py_dependencies += dep_cuda
endif

pyarb = python3.extension_module('_arbor',
    sources: pyarb_srcs,
    dependencies : py_dependencies,
    link_language : 'cpp',
    override_options: [
        'cpp_rtti=true',
    ],
    install: true,
    subdir: 'arbor'
)

python3.install_sources(['__init__.py', '../VERSION'], subdir: 'arbor')

