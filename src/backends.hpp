#pragma once

#include <string>
#include <backends/fvm.hpp>

namespace nest {
namespace mc {

enum class backend_policy {
    multicore,      //  use multicore backend for all computation
    gpu          //  use gpu back end when supported by cell_group type
};

inline std::string to_string(backend_policy p) {
    if (p==backend_policy::multicore) {
        return "multicore";
    }
    return "gpu";
}

} // namespace mc
} // namespace nest
