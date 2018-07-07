#include <arbor/mechcat.hpp>

#include "backends/builtin_mech_proto.hpp"

#include "backends/multicore/fvm.hpp"
#if ARB_HAVE_GPU
#include "backends/gpu/fvm.hpp"
#endif

namespace arb {

template <typename B>
concrete_mech_ptr<B> make_builtin_stimulus();

mechanism_catalogue build_builtin_mechanisms() {
    mechanism_catalogue cat;

    cat.add("_builtin_stimulus", builtin_stimulus_info());

    cat.register_implementation("_builtin_stimulus", make_builtin_stimulus<multicore::backend>());

#if ARB_HAVE_GPU
    cat.register_implementation("_builtin_stimulus", make_builtin_stimulus<gpu::backend>());
#endif

    return cat;
}

const mechanism_catalogue& builtin_mechanisms() {
    static mechanism_catalogue cat = build_builtin_mechanisms();
    return cat;
}

} // namespace arb

