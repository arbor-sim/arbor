#pragma once

#if defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)
//  Intel
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__)
#       define ARB_SYMBOL_IMPORT __attribute__((__dllimport__))
#       define ARB_SYMBOL_EXPORT __attribute__((__dllexport__))
#   elif defined(__GNUC__) && (__GNUC__ >= 4)
#       define ARB_SYMBOL_EXPORT __attribute__((visibility("default")))
#       define ARB_SYMBOL_VISIBLE __attribute__((__visibility__("default")))
#   endif

#elif defined(__clang__)
//  Clang C++
#   if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__)
#       define ARB_SYMBOL_IMPORT __attribute__((__dllimport__))
#       define ARB_SYMBOL_EXPORT __attribute__((__dllexport__))
#   else
#       define ARB_SYMBOL_EXPORT __attribute__((__visibility__("default")))
#       define ARB_SYMBOL_VISIBLE __attribute__((__visibility__("default")))
#   endif

# elif defined(__GNUC__)
//  GNU C++:
#   if __GNUC__ >= 4
#       if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__)
#           define ARB_SYMBOL_IMPORT __attribute__((__dllimport__))
#           define ARB_SYMBOL_EXPORT __attribute__((__dllexport__))
#       else
#           define ARB_SYMBOL_EXPORT __attribute__((__visibility__("default")))
#           define ARB_SYMBOL_VISIBLE __attribute__((__visibility__("default")))
#       endif
#   endif

#endif

#if defined(macintosh) || defined(__APPLE__) || defined(__APPLE_CC__)
// MacOS
#   define ARB_ON_MACOS
#endif

#ifndef ARB_SYMBOL_IMPORT
#   define ARB_SYMBOL_IMPORT
#endif
#ifndef ARB_SYMBOL_EXPORT
#   define ARB_SYMBOL_EXPORT
#endif
#ifndef ARB_SYMBOL_VISIBLE
#   define ARB_SYMBOL_VISIBLE
#endif

