#pragma once

// Specialied type erasure for pointer types.
//
// `any_ptr` represents a non-owning pointer to an arbitrary type
// that can be confirmed at run-time.
//
// Semantics:
//
// 1. An `any_ptr` value p represents either a null pointer, or
//    a non-null pointer of a specific but arbitrary type T.
//
// 2. The value of the pointer as a `void*` value can be retrieved
//    with the member function `as<void*>()`.
//
// 3. The value of the pointer as a pointer of type T is retrieved
//    with the member function `as<T>()`.
//
// 4. When U is not `void *` and is not the type T, the member function
//    will return `nullptr`.
//
// Overloads are provided for `util::any_cast`, with
// `util::any_cast<U>(anyp)` equal to `anyp.as<U>()` for a value
// `anyp` of type `util::any_ptr.
//
// `any_ptr` values are ordered by the value of the corresponding
// `void*` pointer.

#include <cstddef>
#include <type_traits>

#include <arbor/export.hpp>
#include <arbor/util/any_cast.hpp>
#include <arbor/util/lexcmp_def.hpp>

namespace arb {
namespace util {

struct ARB_SYMBOL_VISIBLE any_ptr {
    any_ptr() = default;

    any_ptr(std::nullptr_t) {}

    template <typename T>
    any_ptr(T* ptr):
        ptr_((void *)ptr), type_ptr_(&typeid(T*)) {}

    const std::type_info& type() const noexcept { return *type_ptr_; }

    bool has_value() const noexcept { return ptr_; }

    operator bool() const noexcept { return has_value(); }

    void reset() noexcept { ptr_ = nullptr; }

    void reset(std::nullptr_t) noexcept { ptr_ = nullptr; }

    template <typename T>
    void reset(T* ptr) noexcept {
        ptr_ = (void*)ptr;
        type_ptr_ = &typeid(T*);
    }

    template <typename T, typename = std::enable_if_t<std::is_pointer<T>::value>>
    T as() const noexcept {
        if (std::is_same<T, void*>::value) {
            return (T)ptr_;
        }
        else {
            return typeid(T)==type()? (T)ptr_: nullptr;
        }
    }

    any_ptr& operator=(const any_ptr& other) noexcept {
        type_ptr_ = other.type_ptr_;
        ptr_ = other.ptr_;
        return *this;
    }

    any_ptr& operator=(std::nullptr_t) noexcept {
        reset();
        return *this;
    }

    template <typename T>
    any_ptr& operator=(T* ptr) noexcept {
        reset(ptr);
        return *this;
    }

private:
    void* ptr_ = nullptr;
    const std::type_info* type_ptr_ = &typeid(void);
};

// Order, compare by pointer value:
ARB_DEFINE_LEXICOGRAPHIC_ORDERING_BY_VALUE(any_ptr, (a.as<void*>()), (b.as<void*>()))

// Overload `util::any_cast` for these pointers.
template <typename T>
T any_cast(any_ptr p) noexcept { return p.as<T>(); }

} // namespace util
} // namespace arb
