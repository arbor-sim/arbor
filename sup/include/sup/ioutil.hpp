#pragma once

// Provides:
//
// * mask_stream
//
//   Stream manipulator that enables or disables writing to a stream based on a flag.
//
// * open_or_throw
//
//   Open an fstream, throwing on error. If the 'excl' flag is set, throw a
//   std::runtime_error if the path exists.

#include <iostream>
#include <fstream>
#include <filesystem>

#include <sup/export.hpp>

namespace sup {


template <typename charT, typename traitsT = std::char_traits<charT> >
class basic_null_streambuf: public std::basic_streambuf<charT, traitsT> {
private:
    using streambuf_type = typename std::basic_streambuf<charT, traitsT>;

public:
    using char_type = typename streambuf_type::char_type;
    using int_type = typename streambuf_type::int_type;
    using pos_type = typename streambuf_type::pos_type;
    using off_type = typename streambuf_type::off_type;
    using traits_type = typename streambuf_type::traits_type;

    ~basic_null_streambuf() override = default;

protected:
    std::streamsize xsputn(const char_type*, std::streamsize count) override {
        return count;
    }

    int_type overflow(int_type c) override {
        return traits_type::not_eof(c);
    }
};

class ARB_SUP_API mask_stream {
public:
    explicit mask_stream(bool mask): mask_(mask) {}

    operator bool() const { return mask_; }

    template <typename charT, typename traitsT>
    friend std::basic_ostream<charT, traitsT>&
    operator<<(std::basic_ostream<charT, traitsT>& O, const mask_stream& F) {
        int xindex = get_xindex();

        auto* saved_streambuf = static_cast<std::basic_streambuf<charT, traitsT>*>(O.pword(xindex));

        if (F.mask_ && saved_streambuf) {
            // re-enable by restoring saved streambuf
            O.pword(xindex) = nullptr;
            O.rdbuf(saved_streambuf);
        }
        else if (!F.mask_ && !saved_streambuf) {
            // disable stream but save old streambuf
            O.pword(xindex) = O.rdbuf();
            O.rdbuf(get_null_streambuf<charT, traitsT>());
        }

        return O;
    }

private:
    // our key for retrieve saved streambufs.
    static int get_xindex() {
        static int xindex = std::ios_base::xalloc();
        return xindex;
    }

    template <typename charT, typename traitsT>
    static std::basic_streambuf<charT, traitsT>* get_null_streambuf() {
        static basic_null_streambuf<charT, traitsT> the_null_streambuf;
        return &the_null_streambuf;
    }

    // true => do not filter
    bool mask_;
};

ARB_SUP_API std::fstream open_or_throw(const std::filesystem::path& p, std::ios_base::openmode, bool exclusive);

inline std::fstream open_or_throw(const std::filesystem::path& p, bool exclusive) {
    using std::ios_base;
    return open_or_throw(p, ios_base::in|ios_base::out, exclusive);
}

} // namespace sup

