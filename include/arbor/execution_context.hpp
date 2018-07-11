#pragma once

#include <memory>
#include <string>

#include <arbor/distributed_context.hpp>
#include <arbor/util/pp_util.hpp>
#include <threading/cthread.hpp>

namespace arb {

struct execution_context {
    distributed_context distributed_context_;
    threading::impl::task_system task_system_;

    execution_context(size_t num_threads): task_system_(num_threads) {};
};

}