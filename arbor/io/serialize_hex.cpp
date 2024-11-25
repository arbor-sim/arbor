// Adaptor for hexadecimal output to a std::ostream.

#include <ostream>
#include <string>

// Required for endianness macros:
#include <sys/types.h>

#include "io/serialize_hex.hpp"

namespace arb {
namespace io {

namespace impl {

    enum class endian {
        little = __ORDER_LITTLE_ENDIAN__,
        big = __ORDER_BIG_ENDIAN__,
        native = __BYTE_ORDER__
    };

    std::ostream& operator<<(std::ostream& out, const hex_inline_wrap& h) {
        using std::ptrdiff_t;

        constexpr bool little = endian::native==endian::little;
        ptrdiff_t width = h.width;
        const unsigned char* from = h.from;
        const unsigned char* end = h.from+h.size;
        std::string buf;

        auto emit = [&buf](unsigned char c) {
            const char* digit = "0123456789abcdef";
            buf += digit[(c>>4)&0xf];
            buf += digit[c&0xf];
        };

        constexpr unsigned bufsz = 512;
        unsigned bufmargin = 4*width+1;

        buf.reserve(bufsz);
        while (end-from>width) {
            if (buf.size()+bufmargin>=bufsz) {
                out << buf;
                buf.clear();
            }
            for (ptrdiff_t i = 0; i<width; ++i) {
                emit(little? from[width-i-1]: from[i]);
            }
            from += width;
            buf += ' ';
        }
        for (ptrdiff_t i = 0; i<end-from; ++i) {
            emit(little? from[width-i-1]: from[i]);
        }

        out << buf;
        return out;
    }

} // namespace impl
} // namespace io
} // namespace arb
