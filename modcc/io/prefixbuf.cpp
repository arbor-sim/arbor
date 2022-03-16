#include <iostream>
#include <stack>
#include <string>
#include <vector>

#include "io/prefixbuf.hpp"

namespace io {

// prefixbuf implementation:

std::streamsize prefixbuf::xsputn(const char_type* s, std::streamsize count) {
    std::streamsize written = 0;

    while (count>0) {
        if (bol_) {
            if (prefix_empty_lines_ || s[0]!='\n') {
                inner_->sputn(&prefix[0], prefix.size());
            }
            bol_ = false;
        }

        std::streamsize i = 0;
        while (i<count && s[i]!='\n') {
            ++i;
        }

        if (i<count) { // encountered '\n'
            ++i;
            bol_ = true;
        }

        std::streamsize n = inner_->sputn(s, i);
        written += n;
        if (n<i) {
            break;
        }

        s += i;
        count -= i;
    }

    return written;
}

prefixbuf::int_type prefixbuf::overflow(int_type ch) {
    static int_type eof = traits_type::eof();

    if (ch!=eof) {
        char_type c = (char_type)ch;
        return xsputn(&c, 1)? 0: eof;
    }

    return eof;
}

// setprefix implementation:

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, const setprefix& sp) {
    if (auto pbuf = dynamic_cast<prefixbuf*>(os.rdbuf())) {
        pbuf->prefix = sp.prefix_;
    }

    return os;
}

// indent_manip implementation:

using indent_stack = std::stack<unsigned, std::vector<unsigned>>;

int indent_manip::xindex() {
    static int i = std::ios_base::xalloc();
    return i;
}

static void apply_indent_prefix(std::ios& s, int index) {
    if (auto pbuf = dynamic_cast<prefixbuf*>(s.rdbuf())) {
        indent_stack* stack_ptr = static_cast<indent_stack*>(s.pword(index));
        unsigned tabwidth = s.iword(index);

        unsigned tabs = (!stack_ptr || stack_ptr->empty())? 0: stack_ptr->top();
        pbuf->prefix = std::string(tabs*tabwidth, ' ');
    }
}

static void indent_stack_callback(std::ios_base::event ev, std::ios_base& ios, int index) {
    void*& pword = ios.pword(index);

    switch (ev) {
    case std::ios_base::erase_event:
        if (pword) {
            indent_stack* stack_ptr = static_cast<indent_stack*>(pword);
            delete stack_ptr;
            pword = nullptr;
        }
        break;
    case std::ios_base::copyfmt_event:
        if (pword) {
            // Clone stack:
            indent_stack* stack_ptr = static_cast<indent_stack*>(pword);
            pword = new indent_stack(*stack_ptr);

            // Set prefix if streambuf is a prefixbuf:
            if (auto stream_ptr = dynamic_cast<std::ios*>(&ios)) {
                apply_indent_prefix(*stream_ptr, index);
            }
        }
        break;
    default:
        ;
    }
}

ARB_LIBMODCC_API std::ostream& operator<<(std::ostream& os, indent_manip in) {
    int xindex = indent_manip::xindex();
    void*& pword = os.pword(xindex);
    long& iword = os.iword(xindex);

    if (!pword) {
        os.register_callback(&indent_stack_callback, xindex);
        pword = new indent_stack();
        iword = static_cast<long>(indent_manip::default_tabwidth);
    }

    indent_stack& stack = *static_cast<indent_stack*>(pword);
    switch (in.action_) {
    case indent_manip::pop:
        while (!stack.empty() && in.value_--) {
            stack.pop();
        }
        break;
    case indent_manip::push:
        stack.push(stack.empty()? in.value_: in.value_+stack.top());
        break;
    case indent_manip::settab:
        iword = static_cast<long>(in.value_);
        break;
    }

    apply_indent_prefix(os, xindex);
    return os;
}

} // namespace io
