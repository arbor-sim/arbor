libmodcc_src = [
    'astmanip.cpp',
    'blocks.cpp',
    'errorvisitor.cpp',
    'expression.cpp',
    'functionexpander.cpp',
    'functioninliner.cpp',
    'lexer.cpp',
    'kineticrewriter.cpp',
    'linearrewriter.cpp',
    'module.cpp',
    'parser.cpp',
    'solvers.cpp',
    'symdiff.cpp',
    'symge.cpp',
    'token.cpp',
    'io/prefixbuf.cpp',
    'printer/cexpr_emit.cpp',
    'printer/cprinter.cpp',
    'printer/marks.cpp',
    'printer/gpuprinter.cpp',
    'printer/infoprinter.cpp',
    'printer/printerutil.cpp']

libmodcc = library('modcc',
              sources: [libmodcc_src],
              dependencies: [dep_libfmt])


modcc = executable('modcc',
                   sources: ['modcc.cpp', ],
                   dependencies: dep_tinyopt,
                   link_with: [libmodcc],
                   install: true)
