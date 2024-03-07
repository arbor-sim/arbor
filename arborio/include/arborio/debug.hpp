#pragma once

#include <string>
#include <functional>
#include <vector>

#include <arbor/export.hpp>
#include <arborio/export.hpp>

#include <arbor/morph/segment_tree.hpp>
#include <arbor/morph/morphology.hpp>

namespace arborio {
ARB_ARBORIO_API std::string show(const arb::segment_tree&);
ARB_ARBORIO_API std::string show(const arb::morphology&);
}
