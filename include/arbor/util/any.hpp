#pragma once

#include <memory>
#include <typeinfo>
#include <type_traits>

// Partial implementation of std::any from C++17 standard.
//      http://en.cppreference.com/w/cpp/utility/any
//
// Implements a standard-compliant subset of the full interface.
//
// - Does not avoid dynamic allocation of small objects.
// - Does not implement the in_place_type<T> constructors from the standard.
// - Does not implement the emplace modifier from the standard.

namespace arb {
namespace util {

// Defines a type of object to be thrown by the value-returning forms of
// util::any_cast on failure.
//      http://en.cppreference.com/w/cpp/utility/any/bad_any_cast
class bad_any_cast: public std::bad_cast {
public:
    const char* what() const noexcept override {
        return "bad any cast";
    }
};

class any {
public:
    constexpr any() = default;

    any(const any& other): state_(other.state_->copy()) {}

    any(any&& other) noexcept {
        std::swap(other.state_, state_);
    }

    template <
        typename T,
        typename = std::enable_if_t<!std::is_same<std::decay_t<T>, any>::value>
    >
    any(T&& other) {
        using contained_type = std::decay_t<T>;
        static_assert(std::is_copy_constructible<contained_type>::value,
            "Type of contained object stored in any must satisfy the CopyConstructible requirements.");

        state_.reset(new model<contained_type>(std::forward<T>(other)));
    }

    any& operator=(const any& other) {
        state_.reset(other.state_->copy());
        return *this;
    }

    any& operator=(any&& other) noexcept {
        swap(other);
        return *this;
    }

    template <
        typename T,
        typename = std::enable_if_t<!std::is_same<std::decay_t<T>, any>::value>
    >
    any& operator=(T&& other) {
        using contained_type = std::decay_t<T>;

        static_assert(std::is_copy_constructible<contained_type>::value,
            "Type of contained object stored in any must satisfy the CopyConstructible requirements.");

        state_.reset(new model<contained_type>(std::forward<T>(other)));
        return *this;
    }

    void reset() noexcept {
        state_.reset(nullptr);
    }

    void swap(any& other) noexcept {
        std::swap(other.state_, state_);
    }

    bool has_value() const noexcept {
        return (bool)state_;
    }

    const std::type_info& type() const noexcept {
        return has_value()? state_->type(): typeid(void);
    }

private:
    struct interface {
        virtual ~interface() = default;
        virtual const std::type_info& type() = 0;
        virtual interface* copy() = 0;
        virtual void* pointer() = 0;
        virtual const void* pointer() const = 0;
    };

    template <typename T>
    struct model: public interface {
        ~model() = default;
        model(const T& other): value(other) {}
        model(T&& other): value(std::move(other)) {}

        interface* copy() override { return new model<T>(*this); }
        const std::type_info& type() override { return typeid(T); }
        void* pointer() override { return &value; }
        const void* pointer() const override { return &value; }

        T value;
    };

    std::unique_ptr<interface> state_;

protected:
    template <typename T>
    friend const T* any_cast(const any* operand);

    template <typename T>
    friend T* any_cast(any* operand);

    template <typename T>
    T* unsafe_cast() {
        return static_cast<T*>(state_->pointer());
    }

    template <typename T>
    const T* unsafe_cast() const {
        return static_cast<const T*>(state_->pointer());
    }
};

namespace impl {

template <typename T>
using any_cast_remove_qual = std::remove_cv_t<std::remove_reference_t<T>>;

} // namespace impl

// If operand is not a null pointer, and the typeid of the requested T matches
// that of the contents of operand, a pointer to the value contained by operand,
// otherwise a null pointer.
template<class T>
const T* any_cast(const any* operand) {
    if (operand && operand->type()==typeid(T)) {
        return operand->unsafe_cast<T>();
    }
    return nullptr;
}

template<class T>
T* any_cast(any* operand) {
    if (operand && operand->type()==typeid(T)) {
        return operand->unsafe_cast<T>();
    }
    return nullptr;
}

template<class T>
T any_cast(const any& operand) {
    using U = impl::any_cast_remove_qual<T>;
    static_assert(std::is_constructible<T, const U&>::value,
        "any_cast type can't construct copy of contained object");

    auto ptr = any_cast<U>(&operand);
    if (ptr==nullptr) {
        throw bad_any_cast();
    }
    return static_cast<T>(*ptr);
}

template<class T>
T any_cast(any& operand) {
    using U = impl::any_cast_remove_qual<T>;
    static_assert(std::is_constructible<T, U&>::value,
        "any_cast type can't construct copy of contained object");

    auto ptr = any_cast<U>(&operand);
    if (ptr==nullptr) {
        throw bad_any_cast();
    }
    return static_cast<T>(*ptr);
}

template<class T>
T any_cast(any&& operand) {
    using U = impl::any_cast_remove_qual<T>;
    static_assert(std::is_constructible<T, U>::value,
        "any_cast type can't construct copy of contained object");

    auto ptr = any_cast<U>(&operand);
    if (ptr==nullptr) {
        throw bad_any_cast();
    }
    return static_cast<T>(std::move(*ptr));
}

// Constructs an any object containing an object of type T, passing the
// provided arguments to T's constructor.
//
// This does not exactly follow the standard, which states that
// make_any is equivalent to
//   return std::any(std::in_place_type<T>, std::forward<Args>(args)...);
// i.e. that the contained object should be constructed in place, whereas
// this implementation constructs the object, then moves it into the
// contained object.
// FIXME: rewrite with in_place_type when available.
template <class T, class... Args>
any make_any(Args&&... args) {
    return any(T(std::forward<Args>(args) ...));
}

} // namespace util
} // namespace arb
