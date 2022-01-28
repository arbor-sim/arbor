# Configure the build_catalogue tool for generating external mechanism catalogues.
buildcat_configuration = configuration_data()

buildcat_configuration.set('arb_version', arb_version)
buildcat_configuration.set('prefix_source', meson.project_source_root())
buildcat_configuration.set('prefix', get_option('prefix'))
buildcat_configuration.set('with_gpu', (arb_with_gpu? 'true': 'false'))
buildcat_configuration.set('modcc_flags',
                             '\''+'\', \''.join(arb_modcc_flags)+'\'')
buildcat_configuration.set('private_args',
                             '\''+'\', \''.join(arbor_private_args)+'\'')
buildcat_configuration.set('cuda_private_args',
                             '\''+'\', \''.join(arbor_cuda_private_args)+'\'')
buildcat_configuration.set('catalogue_options',
                             '\''+'\', \''.join(catalogue_options)+'\'')

configure_file(input : 'build-meson-catalogue.in',
               output : 'build-catalogue',
               configuration : buildcat_configuration)


