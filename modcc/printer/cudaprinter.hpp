#pragma once

#include <string>

#include "module.hpp"

std::string emit_cuda_cpp_source(const Module& m, const std::string& ns);
std::string emit_cuda_cu_source(const Module& m, const std::string& ns);
