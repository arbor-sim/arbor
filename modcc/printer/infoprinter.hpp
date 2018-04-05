#pragma once

#include <string>

#include "module.hpp"

// Build header file comprising mechanism metadata
// and declarations of backend-specific mechanism implementations.

std::string build_info_header(const Module& m, const std::string& qual_namespace);

