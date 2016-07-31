#pragma once

#include <iostream>

namespace nest {
namespace mc {
namespace util {

class iosfmt_guard {
public:
    explicit iosfmt_guard(std::ios& stream) :
        save_(nullptr), stream_(stream)
    {
        save_.copyfmt(stream_);
    }

    ~iosfmt_guard() {
        stream_.copyfmt(save_);
    }

private:
    std::ios save_;
    std::ios& stream_;
};


template <typename charT,typename traitsT=std::char_traits<charT> >
struct basic_null_streambuf: std::basic_streambuf<charT,traitsT> {
private:
    typedef typename std::basic_streambuf<charT,traitsT> streambuf_type;

public:
    typedef typename streambuf_type::char_type char_type;
    typedef typename streambuf_type::int_type int_type;
    typedef typename streambuf_type::pos_type pos_type;
    typedef typename streambuf_type::off_type off_type;
    typedef typename streambuf_type::traits_type traits_type;

    virtual ~basic_null_streambuf() {}

protected:
    virtual std::streamsize xsputn(const char_type *s,std::streamsize count) {
        return count;
    }

    virtual int_type overflow(char c) {
        return traits_type::not_eof(c);
    }
};

struct mask_stream {
    explicit mask_stream(bool mask_): mask(mask_) {}

    operator bool() const { return mask; }

    template <typename charT,typename traitsT>
    friend std::basic_ostream<charT,traitsT> &
    operator<<(std::basic_ostream<charT,traitsT> &O,const mask_stream &F) {
        int xindex=get_xindex();

        std::basic_streambuf<charT,traitsT> *saved_streambuf=
            static_cast<std::basic_streambuf<charT,traitsT> *>(O.pword(xindex));

        if (F.mask && saved_streambuf) {
            // re-enable by restoring saved streambuf
            O.pword(xindex)=0;
            O.rdbuf(saved_streambuf);
        }
        else if (!F.mask && !saved_streambuf) {
            // disable stream but save old streambuf
            O.pword(xindex)=O.rdbuf();
            O.rdbuf(get_null_streambuf<charT,traitsT>());
        }

        return O;
    }

private:
    // our key for retrieve saved streambufs.
    static int get_xindex() {
        static int xindex=std::ios_base::xalloc();
        return xindex;
    }

    template <typename charT,typename traitsT>
    static std::basic_streambuf<charT,traitsT> *get_null_streambuf() {
        static basic_null_streambuf<charT,traitsT> the_null_streambuf;
        return &the_null_streambuf;
    }

    // true => do not filter
    bool mask;
};

} // namespace util
} // namespace mc
} // namespace nest

