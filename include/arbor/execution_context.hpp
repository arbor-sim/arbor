#pragma once

#include <memory>
#include <string>

#include <arbor/distributed_context.hpp>
#include <arbor/util/pp_util.hpp>

namespace arb {

struct execution_context {
    distributed_context distributed;
};

}