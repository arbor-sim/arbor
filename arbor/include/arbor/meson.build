# configure headers
# todo: find how to install "assert_macro.hpp"
assert_macro_path=meson.current_build_dir()+'/assert_macro.hpp'
if with_assertions.enabled()
  run_command('cp', 'assert_macro.hpp.enabled', assert_macro_path)
else
  run_command('cp', 'assert_macro.hpp.disabled', assert_macro_path)
endif

conf = configuration_data()

# todo: get this version number either from VERSION file, or from project options
conf.set('version', '0.2.1-dev')

# todo: external call to git: see arbor/include/git-source-id
gitlog = run_command('../gitlog')
conf.set('gitlog', gitlog.stdout().strip())

if with_assertions.enabled()
  conf.set('ARB_ASSERTIONS_ENABLED', true)
endif

configure_file(input : 'version.hpp.in',
               output : 'version.hpp',
               configuration : conf)
# todo: add 'install' and 'install_path' so that version.hpp is installed
