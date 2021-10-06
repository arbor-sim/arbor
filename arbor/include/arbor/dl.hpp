#pragma once

#ifdef __APPLE__
#include dl_platform_posix.hpp
#endif

#ifdef __linux__
#include dl_platform_posix.hpp
#endif

#ifndef DL
#error "No platform with dynamic loading found"
#endif
