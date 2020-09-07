#pragma once

#include <cstddef>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <memory>
#include <type_traits>
#include <vector>

#include <arbor/util/either.hpp>
#include <arbor/util/optional.hpp>

namespace arb {

// Forward iterator that can translate a raw stream to valid s_expr input if,
// perchance, you want to parse a Neurolucida .asc file.
// I am not fond of .asc files, which would be s-expressions, if they
// didn't sometimes contain '|' and ',' characters which translate to ')(' and
// '' respectively.
//
// As a result, we create the following hot mess to leave the s-expression
// parser uncontaminated.
class s_expr_stream {
    using iterator_type = std::string::const_iterator;
    iterator_type pos_;
    iterator_type end_;

    using iterator = s_expr_stream;
    using const_iterator = const s_expr_stream;
    using difference_type = std::string::difference_type;

    bool pipes_ = false;
    bool munch_commas_ = false;
    enum pipe_state {ps_none, ps_lparen, ps_rparen};
    int pipe_state_ = ps_none;

    void set_state() {
        pipe_state_ = ps_none;
        if (pipes_ && *pos_=='|') {
            pipe_state_ = ps_rparen;
            return;
        }

        if (munch_commas_) {
            while (pos_!=end_ && *pos_==',') {
                ++pos_;
            }
        }
    }

    public:

    s_expr_stream(const std::string& s, bool pipes=false, bool munch_commas=false):
        pos_(s.cbegin()),
        end_(s.cend()),
        pipes_(pipes),
        munch_commas_(munch_commas)
    {
        if (pos_!=end_) {
            set_state();
        }
    }

    char operator*() const {
        if (pos_==end_) {
            return '\0';
        }
        switch (pipe_state_) {
            case ps_lparen:
                return '(';
            case ps_rparen:
                return ')';
            default:
                return *pos_;
        }
    }

    s_expr_stream& operator++() {
        // If already at the end don't advance.
        if (pos_==end_) {
            return *this;
        }

        if (pipe_state_==ps_rparen) {
            pipe_state_=ps_lparen;
            return *this;
        }

        ++pos_;

        set_state();
        return *this;
    }

    s_expr_stream operator++(int) {
        s_expr_stream it = *this;

        ++(*this);

        return it;
    }

    s_expr_stream operator+(unsigned n) {
        s_expr_stream it = *this;

        while (n--) ++it;

        return it;
    }

    char peek(unsigned i) {
        return *(*this+i);
    }

    bool operator==(const s_expr_stream& other) const {
        return pos_==other.pos_ && pipe_state_==other.pipe_state_;
    }

    bool operator!=(const s_expr_stream& other) {
        return !(*this==other);
    }

    difference_type operator-(const s_expr_stream& rhs) const {
        return pos_ - rhs.pos_;
    }
};

struct src_location {
    unsigned line = 0;
    unsigned column = 0;

    src_location() = default;

    src_location(unsigned l, unsigned c):
        line(l), column(c)
    {}
};

std::ostream& operator<<(std::ostream& o, const src_location& l);

enum class tok {
    nil,
    real,       // real number
    integer,    // integer
    name,       // name
    lparen,     // left parenthesis '('
    rparen,     // right parenthesis ')'
    string,     // string, written as "spelling"
    eof,        // end of file/input
    error       // special error state marker
};

std::ostream& operator<<(std::ostream&, const tok&);

struct token {
    src_location loc;
    tok kind;
    std::string spelling;
};

std::ostream& operator<<(std::ostream&, const token&);

std::vector<token> tokenize(const char* line);

struct s_expr {
    template <typename U>
    struct s_pair {
        U head = U();
        U tail = U();
        s_pair(U l, U r): head(std::move(l)), tail(std::move(r)) {}
    };

    // This value_wrapper is used to wrap the shared pointer
    template <typename T>
    struct value_wrapper{
        using state_t = std::unique_ptr<T>;
        state_t state;

        value_wrapper() = default;

        value_wrapper(const T& v):
            state(std::make_unique<T>(v)) {}

        value_wrapper(T&& v):
            state(std::make_unique<T>(std::move(v))) {}

        value_wrapper(const value_wrapper& other):
            state(std::make_unique<T>(other.get())) {}

        value_wrapper& operator=(const value_wrapper& other) {
            state = std::make_unique<T>(other.get());
            return *this;
        }

        value_wrapper(value_wrapper&& other) = default;

        friend std::ostream& operator<<(std::ostream& o, const value_wrapper& w) {
            return o << *w.state;
        }

        operator T() const {
            return *state;
        }

        const T& get() const {
            return *state;
        }

        T& get() {
            return *state;
        }
    };

    template <bool Const>
    class s_expr_iterator_impl {
        public:

        struct sentinel {};

        using value_type = s_expr;
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using pointer   = std::conditional_t<Const, const s_expr*, s_expr*>;
        using reference = std::conditional_t<Const, const s_expr&, s_expr&>;

        s_expr_iterator_impl(reference e):
            inner_(&e)
        {
            if (inner_->is_atom()) {
                throw std::runtime_error("Attempt to create s_expr_iterator on an atom.");
            }
            if (finished()) inner_ = nullptr;
        }

        s_expr_iterator_impl(const sentinel& e):
            inner_(nullptr)
        {}

        reference operator*() const {
            return inner_->head();
        }

        pointer operator->() const {
            return &inner_->head();
        }

        s_expr_iterator_impl& operator++() {
            advance();
            return *this;
        }

        s_expr_iterator_impl operator++(int) {
            s_expr_iterator_impl cur = *this;
            advance();
            return cur;
        }

        s_expr_iterator_impl operator+(difference_type i) const {
            s_expr_iterator_impl it = *this;
            while (i--) {
                ++it;
            }
            return it;
        }
        bool operator==(const s_expr_iterator_impl& other) const {
            return inner_==other.inner_;
        }
        bool operator!=(const s_expr_iterator_impl& other) const {
            return !(*this==other);
        }
        bool operator==(const sentinel& other) const {
            return !inner_;
        }
        bool operator!=(const sentinel& other) const {
            return !(*this==other);
        }

        reference expression() const {
            return *inner_;
        }

        private:

        bool finished() const {
            return inner_->is_atom() && inner_->atom().kind==tok::nil;
        }

        void advance() {
            if (!inner_) return;
            inner_ = &inner_->tail();
            if (finished()) inner_ = nullptr;
        }

        // Pointer to the current s_expr.
        // Set to nullptr when at the end of the range.
        pointer inner_;
    };

    using iterator       = s_expr_iterator_impl<false>;
    using const_iterator = s_expr_iterator_impl<true>;

    // An s_expr can be one of
    //      1. an atom
    //      2. a pair of s_expr (head and tail)
    // The s_expr uses a util::either to represent these two possible states,
    // which requires using an incomplete definition of s_expr, requiring
    // with a std::unique_ptr via value_wrapper.

    using pair_type = s_pair<value_wrapper<s_expr>>;
    arb::util::either<token, pair_type> state = token{{0,0}, tok::nil, "nil"};

    s_expr(const s_expr& s): state(s.state) {}
    s_expr() = default;
    s_expr(token t): state(std::move(t)) {}
    s_expr(s_expr l, s_expr r):
        state(pair_type(std::move(l), std::move(r)))
    {}

    bool is_atom() const;

    const token& atom() const;

    operator bool() const;

    const s_expr& head() const;
    const s_expr& tail() const;
    s_expr& head();
    s_expr& tail();

    iterator       begin()        { return {*this}; }
    iterator       end()          { return iterator::sentinel{}; }
    const_iterator begin()  const { return {*this}; }
    const_iterator end()    const { return const_iterator::sentinel{}; }
    const_iterator cbegin() const { return {*this}; }
    const_iterator cend()   const { return const_iterator::sentinel{}; }

    friend std::ostream& operator<<(std::ostream& o, const s_expr& x);
};

std::size_t length(const s_expr& l);
src_location location(const s_expr& l);

s_expr parse(const std::string& line);
s_expr parse(s_expr_stream begin);
std::vector<s_expr> parse_multi(s_expr_stream begin);

} // namespace arb

