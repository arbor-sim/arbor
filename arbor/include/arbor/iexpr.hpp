#pragma once

// Implementations for inhomogeneous expressions.

#include <any>
#include <memory>
#include <string>
#include <ostream>

#include <arbor/export.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>

namespace arb {

struct mprovider;

enum class iexpr_type {
    scalar,
    distance,
    proximal_distance,
    distal_distance,
    interpolation,
    radius,
    diameter,
    add,
    sub,
    mul,
    div,
    exp,
    step_right,
    step_left,
    step,
    log,
    named
};

struct ARB_SYMBOL_VISIBLE iexpr {
    // Convert to scalar expr type
    iexpr(double value);

    iexpr_type type() const { return type_; }

    const std::any& args() const { return args_; }

    static iexpr scalar(double value);

    static iexpr pi();

    static iexpr distance(double scale, locset loc);

    static iexpr distance(locset loc);

    static iexpr distance(double scale, region reg);

    static iexpr distance(region reg);

    static iexpr proximal_distance(double scale, locset loc);

    static iexpr proximal_distance(locset loc);

    static iexpr proximal_distance(double scale, region reg);

    static iexpr proximal_distance(region reg);

    static iexpr distal_distance(double scale, locset loc);

    static iexpr distal_distance(locset loc);

    static iexpr distal_distance(double scale, region reg);

    static iexpr distal_distance(region reg);

    static iexpr interpolation(double prox_value,
        locset prox_list,
        double dist_value,
        locset dist_list);

    static iexpr interpolation(double prox_value,
        region prox_list,
        double dist_value,
        region dist_list);

    static iexpr radius(double scale);

    static iexpr radius();

    static iexpr diameter(double scale);

    static iexpr diameter();

    static iexpr add(iexpr left, iexpr right);

    static iexpr sub(iexpr left, iexpr right);

    static iexpr mul(iexpr left, iexpr right);

    static iexpr div(iexpr left, iexpr right);

    static iexpr exp(iexpr value);

    static iexpr step_right(iexpr value);

    static iexpr step_left(iexpr value);

    static iexpr step(iexpr value);

    static iexpr log(iexpr value);

    static iexpr named(std::string name);


private:
    iexpr(iexpr_type type, std::any args): type_(type), args_(std::move(args)) {}

    iexpr_type type_;
    std::any args_;
};

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const iexpr& e);

ARB_ARBOR_API inline iexpr operator+(iexpr a, iexpr b) { return iexpr::add(std::move(a), std::move(b)); }

ARB_ARBOR_API inline iexpr operator-(iexpr a, iexpr b) { return iexpr::sub(std::move(a), std::move(b)); }

ARB_ARBOR_API inline iexpr operator*(iexpr a, iexpr b) { return iexpr::mul(std::move(a), std::move(b)); }

ARB_ARBOR_API inline iexpr operator/(iexpr a, iexpr b) { return iexpr::div(std::move(a), std::move(b)); }

ARB_ARBOR_API inline iexpr operator+(iexpr a) { return a; }

ARB_ARBOR_API inline iexpr operator-(iexpr a) { return iexpr::mul(-1.0, std::move(a)); }

struct ARB_SYMBOL_VISIBLE iexpr_interface {

    virtual double eval(const mprovider& p, const mcable& c) const = 0;

    virtual ~iexpr_interface() = default;
};

using iexpr_ptr = std::shared_ptr<iexpr_interface>;

ARB_ARBOR_API iexpr_ptr thingify(const iexpr& expr, const mprovider& m);

}  // namespace arb
