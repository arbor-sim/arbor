#pragma once

#include <memory>
#include <string>

#include <arbor/distributed_context.hpp>
#include <arbor/util/pp_util.hpp>
#include <arbor/threadinfo.hpp>


namespace arb {
namespace threading {
    class task_system;
}
using task_system_handle = std::shared_ptr<threading::task_system>;

task_system_handle make_ts (int nthreads);

struct execution_context {
    distributed_context distributed_context_;
    task_system_handle task_system_;

    execution_context(): task_system_(arb::make_ts(arb::num_threads())) {};
    execution_context(size_t num_threads): task_system_(arb::make_ts(num_threads)) {};
};

}
