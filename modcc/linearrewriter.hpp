#pragma once

#include "expression.hpp"
#include <libmodcc/export.hpp>

// Translate a supplied LINEAR block.
ARB_LIBMODCC_API expression_ptr linear_rewrite(BlockExpression*, std::vector<std::string>);
