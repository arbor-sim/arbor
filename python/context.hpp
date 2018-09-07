#pragma once

#include <arbor/context.hpp>

namespace arb {
namespace py {

struct context_shim {
    arb::context context;
    context_shim(arb::context&& c): context(std::move(c)) {}
};

} // namespace py
} // namespace arb
