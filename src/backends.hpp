#pragma once

#include <string>
#include <backends/fvm.hpp>

namespace nest {
namespace mc {

enum class backend_policy {
    use_multicore,      //  use multicore backend for all computation
    prefer_gpu          //  use gpu back end when supported by cell_group type
};

inline std::string to_string(backend_policy p) {
    if (p==backend_policy::use_multicore) {
        return "use_multicore";
    }
    return "prefer_gpu";
}

} // namespace mc
} // namespace nest
