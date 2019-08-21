#pragma once

#include "expression.hpp"

// Translate a supplied LINEAR block.
expression_ptr linear_rewrite(BlockExpression*, std::vector<std::string>);
