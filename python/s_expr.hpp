#pragma once

#include <string>
#include <memory>
#include <vector>

#include <arbor/util/either.hpp>
#include <arbor/util/optional.hpp>

namespace pyarb {

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
    int column;
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

    // An s_expr can be one of
    //      1. an atom
    //      2. a pair of s_expr (head and tail)
    // The s_expr uses a util::either to represent these two possible states,
    // which requires using an incomplete definition of s_expr, requiring
    // with a std::shared_ptr.

    using pair_type = s_pair<value_wrapper<s_expr>>;
    arb::util::either<token, pair_type> state;

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

    friend std::ostream& operator<<(std::ostream& o, const s_expr& x);
};

std::size_t length(const s_expr& l);
int location(const s_expr& l);

s_expr parse(const char* line);
s_expr parse(const std::string& line);

bool test_identifier(const char* in);

} // namespace pyarb

