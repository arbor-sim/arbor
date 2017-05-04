#pragma once

#include <memory>
#include <typeinfo>
#include <type_traits>

#include <util/meta.hpp>

// Partial implementation of std::any from C++17 standard.
//      http://en.cppreference.com/w/cpp/utility/any
//
// Implements a standard-compliant subset of the full interface.
//
// Does not attempt to avoid dynamic allocation of small objects.


namespace nest {
namespace mc {
namespace util {

class any {
public:
    constexpr any() = default;

    any(const any& other):
        state_(other.state_->copy())
    {}

    any(any&& other) {
        std::swap(other.state_, state_);
    }

    // any uses value semantics
    template <
        typename T,
        typename = typename
            util::enable_if_t<!std::is_same<util::decay_t<T>, any>::value>
    >
    any(T&& other):
        state_(new model<util::decay_t<T>>(std::forward<T>(other)))
    {}

    bool has_value() const {
        return (bool)state_;
    }

    const std::type_info& type() const {
        return state_->type();
    }

private:

    struct concept {
        virtual ~concept() = default;

        // TODO: comments
        virtual const std::type_info& type() = 0;
        virtual concept* copy() = 0;
        //virtual void swap(concept& other) = 0;
        //void(*move)(storage_union& src, storage_union& dest) noexcept;
    };

    template <typename T>
    struct model: public concept {
        ~model() = default;

        concept* copy() override {
            return new model<T>(*this);
        }

        const std::type_info& type() override {
            return typeid(T);
        }

        model(const T& other):
            value(other)
        {}

        model(T&& other):
            value(std::move(other))
        {}

        T value;
    };

    std::unique_ptr<concept> state_;
};

template<class T>
T any_cast(const any& operand) {

}

/*
template<class T>
T any_cast(any& operand);

template<class T>
T any_cast(any&& operand);

template<class T>
const T* any_cast(const any* operand);

template<class T>
T* any_cast(any* operand);
*/

} // namespace util
} // namespace mc
} // namespace nest
