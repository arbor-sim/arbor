arborio_src = [
  'asc_lexer.cpp',
  'neurolucida.cpp',
  'swcio.cpp',
  'cableio.cpp',
  'cv_policy_parse.cpp',
  'label_parse.cpp']

dep_arborio_internal = []

if arb_with_neuroml
  arborio_src += ['neuroml.cpp', 'nml_parse_morphology.cpp', 'xml.cpp', 'xmlwrap.cpp']
  dep_arborio_internal += [dep_libxml2]
endif

lib_arborio = library('arborio',
  sources: arborio_src,
  include_directories: 'include',
  dependencies : [dep_arb, dep_arborio_internal],
  cpp_args : arbor_private_args,
  install: true)

install_subdir('include/arborio', install_dir: 'include')

dep_arborio = declare_dependency(
  include_directories: ['include'],
  link_with: lib_arborio,
  dependencies: dep_arborio_internal)
