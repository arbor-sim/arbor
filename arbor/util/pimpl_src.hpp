#pragma once

// Contains pimpl wrapper's implementation which delegates to the actual implementation
// Include this file in the implementation's source file

#include <utility>

#include <util/pimpl.hpp>

namespace arb {
namespace util {

template<typename T>
pimpl<T>::~pimpl() = default;

template<typename T>
pimpl<T>::pimpl() noexcept = default;

template<typename T>
pimpl<T>::pimpl(T* ptr) noexcept : m{ptr} {}

template<typename T>
template<typename... Args>
pimpl<T>::pimpl(Args&&... args)
: m{new T{std::forward<Args>(args)...}} {}

template<typename T>
pimpl<T>::pimpl(pimpl&&) noexcept = default;

template<typename T>
pimpl<T>& pimpl<T>::operator=(pimpl&&) noexcept = default;

template<typename T>
T* pimpl<T>::operator->() noexcept { return m.get(); }

template<typename T>
const T* pimpl<T>::operator->() const noexcept { return m.get(); }

template<typename T>
T& pimpl<T>::operator*() noexcept { return *m.get(); }

template<typename T>
const T& pimpl<T>::operator*() const noexcept { return *m.get(); }

template<typename T, typename... Args>
pimpl<T> make_pimpl(Args&&... args) {
    return {new T{std::forward<Args>(args)...}};
}

} // namespace util
} // namespace arb

// In order to avoid linker errors for the constructors and destructor, the pimpl template needs to
// be instantiated in the source file. This macro helps with this boilerplate code. Note, that it
// needs to be placed in the default namespace.
#define ARB_INSTANTIATE_PIMPL(T) \
namespace arb {                  \
namespace util {                 \
template struct pimpl<T>;        \
}                                \
}
