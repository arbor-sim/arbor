#pragma once

#include "expression.hpp"

// Translate a supplied KINETIC block to equivalent DERIVATIVE block.
expression_ptr linear_rewrite(BlockExpression*, std::vector<std::string>);
