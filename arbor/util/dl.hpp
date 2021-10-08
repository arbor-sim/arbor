#pragma once

#ifdef __APPLE__
#include "dl_platform_posix.hpp"
#endif

#ifdef __linux__
#include "dl_platform_posix.hpp"
#endif

#ifndef DL
#error "No platform support for dynamically loading libraries found."
#endif
