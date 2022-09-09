#pragma once

#include <any>
#include <string>

#include <arbor/arbexcept.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/util/expected.hpp>
#include <arbor/iexpr.hpp>

#include <arbor/s_expr.hpp>
#include <arborio/export.hpp>

namespace arborio {

struct ARB_SYMBOL_VISIBLE label_parse_error: arb::arbor_exception {
    explicit label_parse_error(const std::string& msg, const arb::src_location& loc);
    explicit label_parse_error(const std::string& msg): arb::arbor_exception(msg) {}
};

template <typename T>
using parse_label_hopefully = arb::util::expected<T, label_parse_error>;

ARB_ARBORIO_API parse_label_hopefully<std::any> parse_label_expression(const std::string&);
ARB_ARBORIO_API parse_label_hopefully<std::any> parse_label_expression(const arb::s_expr&);

ARB_ARBORIO_API parse_label_hopefully<arb::region> parse_region_expression(const std::string& s);
ARB_ARBORIO_API parse_label_hopefully<arb::locset> parse_locset_expression(const std::string& s);
ARB_ARBORIO_API parse_label_hopefully<arb::iexpr> parse_iexpr_expression(const std::string& s);

namespace literals {

struct morph_from_string {
    morph_from_string(const std::string& s): str{s} {}
    morph_from_string(const char* s): str{s} {}

    std::string str;

    operator arb::locset() const {
        if (auto r = parse_locset_expression(str)) return *r;
        else throw r.error();
    }

    operator arb::region() const {
        if (auto r = parse_region_expression(str)) return *r;
        else throw r.error();
    }
};

struct morph_from_label {
    morph_from_label(const std::string& s): str{s} {}
    morph_from_label(const char* s): str{s} {}

    std::string str;

    operator arb::locset() const { return arb::ls::named(str); }
    operator arb::region() const { return arb::reg::named(str); }
};

inline
arb::locset operator "" _ls(const char* s, std::size_t) {
    if (auto r = parse_locset_expression(s)) return *r;
    else throw r.error();
}

inline
arb::region operator "" _reg(const char* s, std::size_t) {
    if (auto r = parse_region_expression(s)) return *r;
    else throw r.error();
}

inline morph_from_string operator "" _morph(const char* s, std::size_t) { return {s}; }
inline morph_from_label operator "" _lab(const char* s, std::size_t) { return {s}; }
} // namespace literals
} // namespace arborio
