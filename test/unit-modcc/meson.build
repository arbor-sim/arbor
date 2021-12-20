unit_modcc_sources = [
    # unit tests
    'test_fn_rewriters.cpp',
    'test_lexer.cpp',
    'test_kinetic_rewriter.cpp',
    'test_module.cpp',
    'test_msparse.cpp',
    'test_parser.cpp',
    'test_prefixbuf.cpp',
    'test_printers.cpp',
    'test_removelocals.cpp',
    'test_simd_backend.cpp',
    'test_symdiff.cpp',
    'test_symge.cpp',
    'test_visitors.cpp',
    # unit test driver
    'driver.cpp',
    # utility
    'common.cpp',
    'expr_expand.cpp']

unit_modcc_datadir = '-DDATADIR="'+meson.current_source_dir()+'"'
unit_modcc = executable('unit-modcc',
                   sources: unit_modcc_sources,
                   dependencies: [dep_libmodcc, dep_gtest],
                   cpp_args: unit_modcc_datadir,
                   install: false,
                   build_by_default: false)
