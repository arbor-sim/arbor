#pragma once

#include <any>
#include <memory>
#include <typeinfo>
#include <type_traits>

#include <arbor/export.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/extra_traits.hpp>

// A non copyable variant of std::any. The two main use cases for such a
// container are:
//     1. storing types that are not copyable, and
//     2. ensuring that no copies are made of copyable types stored within a
//        in a type-erased container.
//
// util::unique_any has the same semantics as std::any with the execption of
// copy and copy assignment, which are explicitly forbidden for all contained
// types. Objects stored in util::unique_any also naturally need not be copy
// constructable.
//
// util::any_cast<T> overloads are provided for util::unique_any, with
// semantics analagous to those of std::any_cast and std::any, with copying
// of the contained value permitted if the type of that value permits it.
//
// Examples:
//
//      unique_any<int> a(3);
//      int& ref = any_cast<int&>(a); // Take a reference.
//      ref = 42;                     // Update contained value via reference.
//      int val = any_cast<int>(a);   // Take a copy.
//      assert(val==42);
//
//      // If the underlying type is not copyable, only references may be
//      // taken to the contained value. For a movable but non-copyable
//      // type `nocopy_t`:
//
//      unique_any<nocopy_t> a();
//      nocopy_t& ref = any_cast<nocopy_t&>(a);              // ok
//      const nocopy_t& cref = any_cast<const nocopy_t&>(a); // ok
//      nocopy_t v = any_cast<nocopy_t&&>(std::move(a));     // ok
//      nocopy_t v = any_cast<nocopy_t>(a);   // Not ok: compile-time error.

namespace arb {
namespace util {

class ARB_SYMBOL_VISIBLE unique_any {
public:
    constexpr unique_any() = default;

    unique_any(unique_any&& other) noexcept {
        std::swap(other.state_, state_);
    }

    template <
        typename T,
        typename = std::enable_if_t<!std::is_same_v<remove_cvref_t<T>, unique_any>>,
        typename = std::enable_if_t<!std::is_same_v<remove_cvref_t<T>, std::any>>
    >
    unique_any(T&& other) {
        state_.reset(new model<contained_type<T>>(std::forward<T>(other)));
    }

    unique_any& operator=(unique_any&& other) noexcept {
        swap(other);
        return *this;
    }

    template <
        typename T,
        typename = std::enable_if_t<!std::is_same_v<remove_cvref_t<T>, unique_any>>,
        typename = std::enable_if_t<!std::is_same_v<remove_cvref_t<T>, std::any>>
    >
    unique_any& operator=(T&& other) {
        state_.reset(new model<contained_type<T>>(std::forward<T>(other)));
        return *this;
    }

    void reset() noexcept {
        state_.reset(nullptr);
    }

    void swap(unique_any& other) noexcept {
        std::swap(other.state_, state_);
    }

    bool has_value() const noexcept {
        return (bool)state_;
    }

    const std::type_info& type() const noexcept {
        return has_value()? state_->type(): typeid(void);
    }

private:
    template <typename T>
    using contained_type = std::decay_t<T>;

    struct interface {
        virtual ~interface() = default;
        virtual const std::type_info& type() = 0;
        virtual void* pointer() = 0;
        virtual const void* pointer() const = 0;
    };

    template <typename T>
    struct model: public interface {
        ~model() override = default;
        model(const T& other): value(other) {}
        model(T&& other): value(std::move(other)) {}

        const std::type_info& type() override { return typeid(T); }
        void* pointer() override { return &value; }
        const void* pointer() const override { return &value; }

        T value;
    };

    std::unique_ptr<interface> state_;

protected:
    template <typename T>
    friend const T* any_cast(const unique_any* operand);

    template <typename T>
    friend T* any_cast(unique_any* operand);

    template <typename T>
    T* unsafe_cast() {
        return static_cast<T*>(state_->pointer());
    }

    template <typename T>
    const T* unsafe_cast() const {
        return static_cast<const T*>(state_->pointer());
    }
};

// If operand is not a null pointer, and the typeid of the requested T matches
// that of the contents of operand, a pointer to the value contained by operand,
// otherwise a null pointer.
template<class T>
const T* any_cast(const unique_any* operand) {
    if (operand && operand->type()==typeid(T)) {
        return operand->unsafe_cast<T>();
    }
    return nullptr;
}

// If operand is not a null pointer, and the typeid of the requested T matches
// that of the contents of operand, a pointer to the value contained by operand,
// otherwise a null pointer.
template<class T>
T* any_cast(unique_any* operand) {
    if (operand && operand->type()==typeid(T)) {
        return operand->unsafe_cast<T>();
    }
    return nullptr;
}

template<class T>
T any_cast(const unique_any& operand) {
    using U = remove_cvref_t<T>;
    static_assert(std::is_constructible<T, const U&>::value,
        "any_cast type can't construct copy of contained object");

    auto ptr = any_cast<U>(&operand);
    if (ptr==nullptr) {
        throw std::bad_any_cast();
    }
    return static_cast<T>(*ptr);
}

template<class T>
T any_cast(unique_any& operand) {
    using U = remove_cvref_t<T>;
    static_assert(std::is_constructible<T, U&>::value,
        "any_cast type can't construct copy of contained object");

    auto ptr = any_cast<U>(&operand);
    if (ptr==nullptr) {
        throw std::bad_any_cast();
    }
    return static_cast<T>(*ptr);
}

template<class T>
T any_cast(unique_any&& operand) {
    using U = remove_cvref_t<T>;

    static_assert(std::is_constructible<T, U&&>::value,
        "any_cast type can't construct copy of contained object");

    auto ptr = any_cast<U>(&operand);
    if (ptr==nullptr) {
        throw std::bad_any_cast();
    }
    return static_cast<T>(std::move(*ptr));
}

// Constructs an any object containing an object of type T, passing the
// provided arguments to T's constructor.
template <class T, class... Args>
unique_any make_unique_any(Args&&... args) {
    return unique_any(T(std::forward<Args>(args) ...));
}

} // namespace util
} // namespace arb
