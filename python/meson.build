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

dep_pybind   = dependency('pybind11', version : '>=2.8.1', required : true)

py3_inst = import('python').find_installation('python3')
python3 = dependency('python3')

pyarb = py3_inst.extension_module('arbor',
    sources: pyarb_srcs,
    dependencies : [python3, dep_pybind, dep_arborio, dep_arb],
    link_language : 'cpp',
    override_options: [
        'cpp_rtti=true',
    ]
)
