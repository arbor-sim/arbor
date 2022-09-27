arb_modcc_flags += ['-t', 'cpu']
catalogue_options = ['-A', 'arbor', '-I', '', '-B', 'multicore']
if arb_with_gpu
  arb_modcc_flags += ['-t', 'gpu']
endif

catalogue_sources = []

subdir('default')
subdir('allen')
subdir('bbp')
