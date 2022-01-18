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

# Configure the meson.build script that will be used to generate external
# mechanism catalogues.
mech_meson_configuration = configuration_data()

mech_meson_configuration.set('arb_version', arb_version)
mech_meson_configuration.set('prefix_source', meson.project_source_root())
mech_meson_configuration.set('prefix', get_option('prefix'))
mech_meson_configuration.set('with_gpu', (arb_with_gpu? 'true': 'false'))
mech_meson_configuration.set('modcc_flags',
                             '\''+'\', \''.join(arb_modcc_flags)+'\'')
mech_meson_configuration.set('private_args',
                             '\''+'\', \''.join(arbor_private_args)+'\'')
mech_meson_configuration.set('cuda_private_args',
                             '\''+'\', \''.join(arbor_cuda_private_args)+'\'')
mech_meson_configuration.set('catalogue_options',
                             '\''+'\', \''.join(catalogue_options)+'\'')

configure_file(input : 'meson.build.in',
               output : 'meson.build',
               configuration : mech_meson_configuration)
