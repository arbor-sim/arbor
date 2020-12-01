#!/usr/bin/env python3

import subprocess as sp
import sys
from tempfile import TemporaryDirectory
import os
from pathlib import Path
import shutil
import stat
import string
import argparse

def parse_arguments():
    def append_slash(s):
        return s+'/' if s and not s.endswith('/') else s

    class ConciseHelpFormatter(argparse.HelpFormatter):
        def __init__(self, **kwargs):
            super(ConciseHelpFormatter, self).__init__(max_help_position=20, **kwargs)

        def _format_action_invocation(self, action):
            if not action.option_strings:
                return super(ConciseHelpFormatter, self)._format_action_invocation(action)
            else:
                optstr = ', '.join(action.option_strings)
                if action.nargs==0:
                    return optstr
                else:
                    return optstr+' '+self._format_args(action, action.dest.upper())

    parser = argparse.ArgumentParser(
        description = 'Generate dynamic catalogue and build it into a shared object.',
        usage = '%(prog)s [options] [module...]',
        add_help = False,
        formatter_class = ConciseHelpFormatter)

    group = parser.add_argument_group('Options')

    group.add_argument(
        '-I', '--module-prefix',
        default = 'mechanisms',
        metavar = 'PATH',
        dest = 'modpfx',
        type = append_slash,
        help = 'directory prefix for module includes, default "%(default)s"')

    group.add_argument(
        '-S', '--arbor-source',
        default = '',
        metavar = 'PATH',
        dest = 'arbsrc',
        type = append_slash,
        help = 'directory prefix for arbor source tree, default "%(default)s"')

    group.add_argument(
        '-A', '--arbor-prefix',
        default = '',
        metavar = 'PATH',
        dest = 'arbpfx',
        type = append_slash,
        help = 'directory prefix for arbor includes, default "%(default)s"')

    group.add_argument(
        '-B', '--backend',
        default = [],
        action = 'append',
        dest = 'backends',
        metavar = 'BACKEND',
        help = 'register implementations for back-end %(metavar)s')

    group.add_argument(
        '-N', '--namespace',
        default = ['arb::catalogue'],
        action = 'append',
        dest = 'namespaces',
        metavar = 'NAMESPACE',
        help = 'add %(metavar)s to list of implicitly included namespaces')

    group.add_argument(
        '-C', '--catalogue',
        default = 'default',
        dest = 'catalogue',
        help = 'catalogue name, default "%(default)s"')

    group.add_argument(
        '-h', '--help',
        action = 'help',
        help = 'display this help and exit')

    return vars(parser.parse_args())

cmake = r"""cmake_minimum_required(VERSION 3.19)

project(catalogue LANGUAGES CXX)

find_package(arbor REQUIRED)

set(external_modcc MODCC "/usr/local/bin/modcc")
if(ARB_WITH_EXTERNAL_MODCC)
    set(external_modcc MODCC ${{modcc}})
endif()

include(BuildModules.cmake)

set(name {name})
set(target ${{name}}_catalogue)
set(mechanisms {mods})

set(src_dir ${{name}})
set(out_dir ${{CMAKE_CURRENT_BINARY_DIR}}/${{name}})
set(cat_src ${{out_dir}}/catalogue.cpp)

set(CMAKE_CXX_COMPILER ${{ARB_CXX}})
set(CMAKE_CXX_FLAGS ${{ARB_CXX_FLAGS}} ${{ARB_CXXOPT_ARCH}})

set(inc "/usr/local/include")

message(STATUS "FLAGS=${{CMAKE_CXX_FLAGS}}")

file(MAKE_DIRECTORY "${{out_dir}}")

build_modules(${{mechanisms}}
              SOURCE_DIR "${{src_dir}}"
              DEST_DIR "${{out_dir}}"
              ${{external_modcc}}
              MODCC_FLAGS -t cpu -t gpu ${{ARB_MODCC_FLAGS}} -N arb::catalogue
              GENERATES .hpp _cpu.cpp _gpu.cpp _gpu.cu
              TARGET build_catalogue_mods)

foreach(mech ${{mechanisms}})
    list(APPEND cat_src ${{out_dir}}/${{mech}}_cpu.cpp)
    if(ARB_WITH_GPU)
        list(APPEND cat_src ${{out_dir}}/${{mech}}_gpu.cpp)
        list(APPEND cat_src ${{out_dir}}/${{mech}}_gpu.cu)
    endif()
endforeach()

set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

add_library(${{target}} SHARED ${{cat_src}})
set_property(TARGET ${{target}} PROPERTY CXX_STANDARD 17)
# this is a terrible hack to get `backends' into scope
target_include_directories(${{target}} PUBLIC {arb_src}/arbor)
# bring generated source into scope
target_include_directories(${{target}} PUBLIC ${{out_dir}})
target_link_libraries(${{target}} PRIVATE arbor::arbor)
"""

build = r"""include(CMakeParseArguments)

# If a MODCC executable is explicitly provided, don't make the in-tree modcc a dependency.

function(build_modules)
    cmake_parse_arguments(build_modules "" "MODCC;TARGET;SOURCE_DIR;DEST_DIR;MECH_SUFFIX" "MODCC_FLAGS;GENERATES" ${ARGN})

    if("${build_modules_SOURCE_DIR}" STREQUAL "")
        set(build_modules_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    if("${build_modules_DEST_DIR}" STREQUAL "")
        set(build_modules_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    file(MAKE_DIRECTORY "${build_modules_DEST_DIR}")

    set(all_generated)
    foreach(mech ${build_modules_UNPARSED_ARGUMENTS})
        set(mod "${build_modules_SOURCE_DIR}/${mech}.mod")
        set(out "${build_modules_DEST_DIR}/${mech}")
        set(generated)
        foreach (suffix ${build_modules_GENERATES})
            list(APPEND generated ${out}${suffix})
        endforeach()

        set(depends "${mod}")
        if(build_modules_MODCC)
            set(modcc_bin ${build_modules_MODCC})
        else()
            list(APPEND depends modcc)
            set(modcc_bin $<TARGET_FILE:modcc>)
        endif()

        set(flags ${build_modules_MODCC_FLAGS} -o "${out}")
        if(build_modules_MECH_SUFFIX)
            list(APPEND flags -m "${mech}${build_modules_MECH_SUFFIX}")
        endif()

        add_custom_command(
            OUTPUT ${generated}
            DEPENDS ${depends}
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMAND ${modcc_bin} ${flags} ${mod}
            COMMENT "modcc generating: ${generated}"
        )
        set_source_files_properties(${generated} PROPERTIES GENERATED TRUE)
        list(APPEND all_generated ${generated})
    endforeach()

    # Fake target to always trigger .mod -> .hpp/.cu dependencies because CMake
    if (build_modules_TARGET)
        set(depends ${all_generated})
        if(NOT build_modules_MODCC)
            list(APPEND depends modcc)
        endif()
        add_custom_target(${build_modules_TARGET} DEPENDS ${depends})
    endif()
endfunction()
"""

cat_src = string.Template(r'''// Automatically generated by:
// $cmdline

#include <${arbpfx}mechcat.hpp>
$backend_includes
$module_includes
$using_namespace

namespace arb {

mechanism_catalogue build_${catalogue}_catalogue() {
    mechanism_catalogue cat;

    $add_modules
    $register_modules
    return cat;
}

const mechanism_catalogue& global_${catalogue}_catalogue() {
    static mechanism_catalogue cat = build_${catalogue}_catalogue();
    return cat;
}

} // namespace arb

extern "C" {
    const arb::mechanism_catalogue& get_catalogue() {
        static auto cat = arb::build_allen_catalogue();
        return cat;
    }
}
''')


def generate(catalogue, modpfx='', arbpfx='', modules=[], backends=[], namespaces=[], **rest):
    def indent(n, lines):
        return '{{:<{0!s}}}'.format(n+1).format('\n').join(lines)

    result = cat_src.safe_substitute(dict(
        cmdline=" ".join(sys.argv),
        arbpfx=arbpfx,
        catalogue=catalogue,
        using_namespace = indent(0,
            ['using namespace {};'.format(n) for n in namespaces]),
        backend_includes = indent(0,
            ['#include "backends/{}/fvm.hpp"'.format(b) for b in backends]),
        module_includes = indent(0,
            ['#include "{}.hpp"'.format(m) for m in modules]),
        add_modules = indent(4,
            ['cat.add("{0}", mechanism_{0}_info());'.format(m) for m in modules]),
        register_modules = indent(4,
            ['cat.register_implementation("{0}", make_mechanism_{0}<{1}::backend>());'.format(m, b)
             for m in modules for b in backends])
        ))
    return result

args = parse_arguments()

name    = args['catalogue']
mod_dir = Path(args['modpfx'])
arb_src = Path(args['arbsrc'])
args['modules'] = [ f[:-4] for f in os.listdir(mod_dir) if f.endswith('.mod') ]

code = generate(**args)

pwd = Path.cwd()

with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    shutil.copytree(pwd / mod_dir, tmp / mod_dir)
    os.mkdir(tmp / 'build')
    os.mkdir(tmp / 'build' / mod_dir)
    os.chdir(tmp)
    with open('BuildModules.cmake', 'w') as fd:
        fd.write(build)
    print(arb_src)
    with open(tmp / 'CMakeLists.txt', 'w') as fd:
        fd.write(cmake.format(name=name, mods=' '.join(args['modules']), arb_src=arb_src))
    with open(tmp / 'build' / mod_dir / 'catalogue.cpp', 'w') as fd:
        fd.write(code)
    os.chdir('build')
    sp.run('cmake ..', shell=True)
    sp.run('make VERBOSE=1', shell=True)
    shutil.copy2(f'{name}_catalogue.so', pwd)
