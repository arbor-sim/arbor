#pragma once

#include <string>

namespace arb {

enum class backend_kind {
    multicore,   //  use multicore backend for all computation
    gpu          //  use gpu back end when supported by cell_group type
};

inline std::string to_string(backend_kind p) {
    switch (p) {
        case backend_kind::multicore:
            return "multicore";
        case backend_kind::gpu:
            return "gpu";
    }
    return "unknown";
}

} // namespace arb
