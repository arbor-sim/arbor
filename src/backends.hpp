#pragma once

#include <string>
#include <backends/fvm.hpp>

namespace nest {
namespace mc {

enum class backend_kind {
    multicore,      //  use multicore backend for all computation
    gpu          //  use gpu back end when supported by cell_group type
};

inline std::string to_string(backend_kind p) {
    if (p==backend_kind::multicore) {
        return "multicore";
    }
    return "gpu";
}

} // namespace mc
} // namespace nest
