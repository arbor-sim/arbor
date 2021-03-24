#include <cerrno>
#include <charconv>
#include <cstdarg>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include <libxml/xmlerror.h>

#include "xmlwrap.hpp"

namespace arborio {
namespace xmlwrap {

namespace detail {

// Note: widely missing library support for floating point std::from_chars.

template <typename V>
bool from_cstr_(V& out, const char* s) {
    auto [p, ec] = std::from_chars(s, s+std::strlen(s), out);
    return ec==std::errc{} && !*p;
}

template <typename V>
std::string nl_to_string_(const V& v, unsigned digits_estimate) {
    std::vector<char> digits(digits_estimate);
    for (;;) {
        if (auto [p, ec] = std::to_chars(digits.data(), digits.data()+digits.size(), v); ec == std::errc{}) {
            return std::string(digits.data(), p);
        }

        digits_estimate *= 2;
        digits.resize(digits_estimate);
    }
}

} // namespace detail


bool nl_from_cstr(std::string& out, const char* content) {
    out = content;
    return true;
}

bool nl_from_cstr(long long& out, const char* content) {
    return detail::from_cstr_(out, content);
}

bool nl_from_cstr(non_negative& out, const char* content) {
    return detail::from_cstr_(out, content);
}

bool nl_from_cstr(double& out, const char* content) {
    // Note: library support is widely missing for floating point std::from_chars,
    // so can't just do:
    //     return detail::from_cstr_(out, content);
    //
    // std::strtod() will use the current C locale, so that's out: anticipating the
    // decimal point character is a race condition.

    std::istringstream is{std::string(content)};
    is.imbue(std::locale::classic());

    double x;
    is >> x;
    if (!is || !is.eof()) return false;
    out = x;
    return true;
}

std::string nl_to_string(non_negative n) {
    return detail::nl_to_string_(n, std::numeric_limits<non_negative>::digits10);
}

std::string nl_to_string(long long n) {
    return detail::nl_to_string_(n, 1+std::numeric_limits<long long>::digits10);
}

void throw_on_xml_generic_error(void *, const char* msg, ...) {
    va_list va, vb;
    va_start(va, msg);
    va_copy(vb, va);

    int r = vsnprintf(nullptr, 0, msg, va);
    va_end(va);

    std::string err(r+1, '\0');
    vsnprintf(&err[0], err.size(), msg, vb);
    va_end(vb);

    throw ::arborio::xml_error(err);
}

void throw_on_xml_structured_error(void *ctx, xmlErrorPtr errp) {
    if (errp->level!=1) { // ignore warnings!
        std::string msg(errp->message);
        if (!msg.empty() && msg.back()=='\n') msg.pop_back();
        throw ::arborio::xml_error(msg, errp->line);
    }
}

xml_error_scope::xml_error_scope() {
    generic_handler_ = xmlGenericError;
    generic_context_ = xmlGenericErrorContext;

    structured_handler_ = xmlStructuredError;
    structured_context_ = xmlStructuredErrorContext;

    xmlSetGenericErrorFunc(nullptr, &throw_on_xml_generic_error);
    xmlSetStructuredErrorFunc((void*)this, &throw_on_xml_structured_error);
}

xml_error_scope::~xml_error_scope() {
    xmlGenericError = generic_handler_;
    xmlGenericErrorContext = generic_context_;

    xmlStructuredError = structured_handler_;
    xmlStructuredErrorContext = structured_context_;
}

} // namespace xmlwrap
} // namespace arborio
