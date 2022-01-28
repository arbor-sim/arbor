gen_cat = find_program('generate_catalogue')
arb_modcc_flags += ['-t', 'cpu', '-t', 'gpu']
catalogue_options = ['-A', 'arbor', '-I', '', '-B', 'multicore']
if arb_with_gpu
  catalogue_options += ['-B', 'gpu']
endif

catalogue_sources = []

subdir('default')
subdir('allen')
subdir('bbp')

install_data('generate_catalogue', install_dir: get_option('bindir'))
