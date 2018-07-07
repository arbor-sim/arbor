#pragma once

// Adaptor for hexadecimal output to a std::ostream.

#include <iostream>

namespace arb {
namespace io {

namespace impl {
    // Wrapper for emitting values on an ostream as a sequence of hex digits.
    struct hex_inline_wrap {
        const unsigned char* from;
        std::size_t size;
        unsigned width;
    };

    std::ostream& operator<<(std::ostream&, const hex_inline_wrap&);
} // namespace impl

// Inline hexadecimal adaptor: group output in `width` bytes.

template <typename T>
impl::hex_inline_wrap hex_inline(const T& obj, unsigned width = 4) {
    return impl::hex_inline_wrap{reinterpret_cast<const unsigned char*>(&obj), sizeof obj, width};
}

// Inline hexadecimal adaptor: print `n` bytes of data from `ptr`, grouping output in `width` bytes.

template <typename T>
impl::hex_inline_wrap hex_inline_n(const T* ptr, std::size_t n, unsigned width = 4) {
    return impl::hex_inline_wrap{reinterpret_cast<const unsigned char*>(ptr), n, width};
}

} // namespace io
} // namespace arb

