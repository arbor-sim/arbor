#pragma once

#include <arbor/arbexcept.hpp>
#include <arbor/util/either.hpp>

namespace arb {
namespace util {

template <typename T, typename E>
struct hopefully {
    using value_type = T;
    using error_type = E;
    arb::util::either<value_type, error_type> state;

    hopefully(const hopefully&) = default;

    hopefully(value_type x): state(std::move(x)) {}
    hopefully(error_type x): state(std::move(x)) {}

    const value_type& operator*() const {
        return try_get();
    }
    value_type& operator*() {
        return try_get();
    }
    const value_type* operator->() const {
        return &try_get();
    }
    value_type* operator->() {
        return &try_get();
    }

    operator bool() const {
        return (bool)state;
    }

    const error_type& error() const {
        try {
            return state.template get<1>();
        }
        catch(arb::util::either_invalid_access& e) {
            throw arb::arbor_internal_error("Attempt to get an error from a valid hopefully wrapper.");
        }
    }

private:

    const value_type& try_get() const {
        try {
            return state.template get<0>();
        }
        catch(arb::util::either_invalid_access& e) {
            throw arb::arbor_internal_error("Attempt to unwrap a hopefully with error state '"+std::string(error().what())+"'");
        }
    }
    value_type& try_get() {
        try {
            return state.template get<0>();
        }
        catch(arb::util::either_invalid_access& e) {
            throw arb::arbor_internal_error("Attempt to unwrap a hopefully with error state '"+std::string(error().what())+"'");
        }
    }
};

} // namespace util
} // namespace arb
