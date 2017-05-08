#pragma once

#include <memory>
#include <typeinfo>
#include <type_traits>

#include <util/any.hpp>
#include <util/meta.hpp>

// A version of util::any that is not copyable.
// The two main use cases for such a container are
//  1. for storing types that are not copyable.
//  2. for ensuring that no copies are made of types that are copyable.
//     e.g. in performance critical code.

namespace nest {
namespace mc {
namespace util {

class unique_any {
public:
    constexpr unique_any() = default;

    unique_any(unique_any&& other) noexcept {
        std::swap(other.state_, state_);
    }

    template <
        typename T,
        typename = typename util::enable_if_t<!std::is_same<util::decay_t<T>, unique_any>::value>
    >
    unique_any(T&& other) {
        using contained_type = util::decay_t<T>;
        state_.reset(new model<contained_type>(std::forward<T>(other)));
    }

    unique_any& operator=(unique_any&& other) noexcept {
        swap(other);
        return *this;
    }

    template <
        typename T,
        typename = typename util::enable_if_t<!std::is_same<util::decay_t<T>, unique_any>::value>
    >
    unique_any& operator=(T&& other) {
        using contained_type = util::decay_t<T>;
        state_.reset(new model<contained_type>(std::forward<T>(other)));
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
    struct interface {
        virtual ~interface() = default;
        virtual const std::type_info& type() = 0;
        virtual void* pointer() = 0;
        virtual const void* pointer() const = 0;
    };

    template <typename T>
    struct model: public interface {
        ~model() = default;

        model(const T& other): value(other) {}

        model(T&& other): value(std::move(other)) {}

        const std::type_info& type() override {
            return typeid(T);
        }

        void* pointer() override {
            return &value;
        }

        const void* pointer() const override {
            return &value;
        }

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

// The semantics of any_cast for unique_any differ from any, because 
// any_cast(any& / const any&/ any&&) return a copy of the contained object
// in the argument to any_cast, while unique_any is designed to work with
// non-copyable types.
// Hence, references to the contained objects are returned, not copies.

template<class T>
T any_cast(const unique_any& operand) {
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
T any_cast(unique_any& operand) {
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
T any_cast(unique_any&& operand) {
    using U = impl::any_cast_remove_qual<T>;

    static_assert(std::is_constructible<T, U&&>::value,
        "any_cast type can't construct copy of contained object");

    auto ptr = any_cast<U>(&operand);
    if (ptr==nullptr) {
        throw bad_any_cast();
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
} // namespace mc
} // namespace nest
