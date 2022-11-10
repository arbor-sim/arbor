#pragma once

#include <memory>

namespace arb {
namespace util {

// A simple wrapper for the pimpl idom inspired by Herb Sutter's GotW #101
template<typename T>
class pimpl
{
  private:
    std::unique_ptr<T> m;

  public:
    ~pimpl();
    
    pimpl() noexcept;

    pimpl(T* ptr) noexcept;

    template<typename... Args>
    pimpl(Args&&... args);

    pimpl(pimpl const&) = delete;
    pimpl(pimpl&&) noexcept;

    pimpl& operator=(pimpl const&) = delete;
    pimpl& operator=(pimpl&&) noexcept;

    T*       operator->() noexcept;
    const T* operator->() const noexcept;
    T&       operator*() noexcept;
    const T& operator*() const noexcept;

    operator bool() const noexcept { return (bool)m; }
};


template<typename T, typename... Args>
pimpl<T> make_pimpl(Args&&... args);

} // namespace util
} // namespace arb
