from distutils.core import setup, Extension
from Cython.Build import cythonize


ext = Extension(
            "py_miniarbor",                 # the extension name
            sources=["py_miniarbor.pyx", "/Users/shaoyc/Github/nestmc-proto/miniapp/miniapp.cpp"],  # additional source file(s)
            include_dirs=["/Users/shaoyc/Github/nestmc-proto", "/Users/shaoyc/Github/nestmc-proto/src", "/Users/shaoyc/Github/nestmc-proto/miniapp"],
            language="c++",             # generate C++ code
            extra_compile_args=["-std=c++11"],
            extra_link_args=["-std=c++11"],

      )

setup(ext_modules=cythonize(ext))




