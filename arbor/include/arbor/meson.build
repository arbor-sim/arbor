# generate the headers:
#   * include/arbor/version.hpp
#   * include/arbor/assert_macro.hpp

fs = import('fs')

# The headers are not regenerated if they already exist, otherwise they would be
# regenerated every time a change is made to any meson.build file in the
# project, which in turn would force all of their dependencies to be rebuilt.
# If a change is made to the meson build description or configuration that
# would generate a different header, the headers must first be deleted.

assert_macro_path=meson.current_build_dir()+'/assert_macro.hpp'
if not fs.is_file(assert_macro_path)
  if arb_with_assertions
    run_command('cp', 'assert_macro.hpp.enabled', assert_macro_path, check: true)
  else
    run_command('cp', 'assert_macro.hpp.disabled', assert_macro_path, check: true)
  endif
endif

version_path=meson.current_build_dir()+'/version.hpp'
if not fs.is_file(version_path)
  conf = configuration_data()

  conf.set('version', arb_version)
  conf.set('version_major', arb_version_major)
  conf.set('version_minor', arb_version_minor)
  conf.set('version_patch', arb_version_patch)
  conf.set('version_dev', arb_version_dev)

  # set all of the #defined values
  foreach opt: arb_configurations
    conf.set(f'ARB_@opt@_ENABLED', true)
  endforeach
  conf.set('ARB_VERSION_MAJOR', arb_version_major)
  conf.set('ARB_VERSION_MINOR', arb_version_minor)
  conf.set('ARB_VERSION_PATCH', arb_version_patch)
  conf.set_quoted('ARB_VERSION_DEV', arb_version_dev)

  gitlog = run_command('../gitlog', check: true).stdout().strip()
  buildtype = get_option('buildtype').to_upper()
  configstring = ';'.join(arb_configurations)
  conf.set_quoted('ARB_VERSION', arb_version)
  conf.set_quoted('ARB_SOURCE_ID', gitlog)
  conf.set_quoted('ARB_ARCH', arb_arch)
  conf.set_quoted('ARB_BUILD_CONFIG', buildtype)
  build_id = f'source_id=@gitlog@;version=@arb_version@;arch=@arb_arch@;config=@buildtype@;@configstring@'
  conf.set_quoted('ARB_FULL_BUILD_ID', build_id)

  configure_file(input : 'version.hpp.in',
                 output : 'version.hpp',
                 configuration : conf)
endif

configure_file(input: meson.global_source_root() + '/cmake/export.hpp.in',
               output: 'export.hpp',
               configuration: {
               'target_short_NAME': 'ARBOR',
               'target': 'arbor',
               'target_name': 'arbor',
               'target_type': f'@arb_build_type@_LIBRARY',
  })

install_headers([assert_macro_path, version_path], install_dir: 'include/arbor')

