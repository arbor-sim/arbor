libmodcc_src = [
    './astmanip.cpp',
    './blocks.cpp',
    './errorvisitor.cpp',
    './expression.cpp',
    './functionexpander.cpp',
    './functioninliner.cpp',
    './kineticrewriter.cpp',
    './lexer.cpp',
    './linearrewriter.cpp',
    './module.cpp',
    './parser.cpp',
    './procinliner.cpp',
    './solvers.cpp',
    './symdiff.cpp',
    './symge.cpp',
    './token.cpp',
    './io/prefixbuf.cpp',
    './printer/cexpr_emit.cpp',
    './printer/cprinter.cpp',
    './printer/gpuprinter.cpp',
    './printer/infoprinter.cpp',
    './printer/marks.cpp',
    './printer/printerutil.cpp']

subdir('libmodcc')

libmodcc = library('modcc',
    sources: [libmodcc_src],
    dependencies: [dep_libfmt],
    include_directories: ['.'],
    install: true,
    cpp_args : '-DFMT_HEADER_ONLY')

dep_libmodcc = declare_dependency(
    include_directories: ['.'],
    link_with: [libmodcc])

modcc = executable('modcc',
     sources: ['modcc.cpp', ],
     dependencies: [dep_tinyopt, dep_libfmt],
     link_with: [libmodcc],
     install: true)
