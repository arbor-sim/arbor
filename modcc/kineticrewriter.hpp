#pragma once

#include "expression.hpp"

// Translate a supplied KINETIC block to equivalent DERIVATIVE block.
expression_ptr kinetic_rewrite(BlockExpression*);
