name = 'arborio'

env_export = configure_file(input: meson.global_source_root() + '/cmake/export.hpp.in',
               output: 'export.hpp',
               configuration: {
               'target_short_NAME': name.to_upper(),
               'target': name,
               'target_name': name,
               'target_type': f'@arb_build_type@_LIBRARY',})
