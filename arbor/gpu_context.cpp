#include <memory>

#include "gpu_context.hpp"

namespace arb {

std::shared_ptr<gpu_context> make_gpu_context() {
    return std::make_shared<gpu_context>();
}

}
