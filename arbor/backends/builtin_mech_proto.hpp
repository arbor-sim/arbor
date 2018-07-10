#pragma once

#include <arbor/mechanism.hpp>
#include <arbor/mechinfo.hpp>

namespace arb {

// Stimulus

inline const mechanism_info& builtin_stimulus_info() {
    using spec = mechanism_field_spec;
    static mechanism_info info = {
        // globals
        {},
        // parameters
        {
            {"delay",     {spec::parameter, "ms", 0, 0}},
            {"duration",  {spec::parameter, "ms", 0, 0}},
            {"amplitude", {spec::parameter, "nA", 0, 0}}
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
