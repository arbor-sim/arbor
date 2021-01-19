#include <arbor/mechcat.hpp>

#include "backends/builtin_mech_proto.hpp"

#include "backends/multicore/fvm.hpp"
#if ARB_HAVE_GPU
#include "backends/gpu/fvm.hpp"
#endif

namespace arb {

mechanism_catalogue build_builtin_mechanisms() {
    // We currently have no builtins! (Stimulus has been removed.)
    return mechanism_catalogue{};
}

const mechanism_catalogue& builtin_mechanisms() {
    static mechanism_catalogue cat = build_builtin_mechanisms();
    return cat;
}

} // namespace arb

