env_export = configure_file(input: meson.global_source_root() + '/cmake/export.hpp.in',
               output: 'export.hpp',
               configuration: {
               'target_short_NAME': 'ARBORENV',
               'target': 'arborenv',
               'target_name': 'arborenv',
               'target_type': 'STATIC', # TODO: figure out shared/static
})
