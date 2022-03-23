#pragma once

#include "expression.hpp"
#include <libmodcc/export.hpp>

// Translate a supplied KINETIC block to equivalent DERIVATIVE block.
ARB_LIBMODCC_API expression_ptr kinetic_rewrite(BlockExpression*);
