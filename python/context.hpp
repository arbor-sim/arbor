#pragma once

#include <arbor/context.hpp>

namespace pyarb {

struct context_shim {
    arb::context context;
    context_shim(arb::context&& c): context(std::move(c)) {}
};

} // namespace pyarb
