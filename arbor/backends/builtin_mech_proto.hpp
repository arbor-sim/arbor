#pragma once

#include <mechanism.hpp>
#include <mechinfo.hpp>

namespace arb {

// Stimulus

inline const mechanism_info& builtin_stimulus_info() {
    using spec = mechanism_field_spec;
    static mechanism_info info = {
        // globals
        {},
        // parameters
        {
            {"delay",     spec(spec::parameter, "ms", 0, 0)},
            {"duration",  spec(spec::parameter, "ms", 0, 0)},
            {"amplitude", spec(spec::parameter, "nA", 0, 0)}
        },
        // state
        {},
        // ions
        {},
        // fingerprint
        "##builtin_stimulus"
    };

    return info;
};

template <typename B>
concrete_mech_ptr<B> make_builtin_stimulus();

} // namespace arb
