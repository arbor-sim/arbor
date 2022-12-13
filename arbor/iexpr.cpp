// Implementations for inhomogeneous expressions.

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>
#include <sstream>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/iexpr.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/util/any_visitor.hpp>
#include <arbor/math.hpp>

namespace arb {

namespace iexpr_impl {
namespace {

msize_t common_parent_branch(msize_t branch_a, msize_t branch_b, const morphology& m) {
    // Locations on different branches.
    // Find first common parent branch. Branch id of parent is
    // always smaller.
    while (branch_a != branch_b) {
        if (branch_b == mnpos || (branch_a != mnpos && branch_a > branch_b))
            branch_a = m.branch_parent(branch_a);
        else
            branch_b = m.branch_parent(branch_b);
    }

    return branch_a;
}

// compute the distance between any two points on the same morphology
double compute_distance(const mlocation& loc_a, const mlocation& loc_b, const mprovider& p) {
    if (loc_a.branch == loc_b.branch) return std::abs(p.embedding().integrate_length(loc_a, loc_b));

    // If mnpos, locations are on different sides of root. Take
    // distance to root in this case. Otherwise, take distance to
    // end of parent branch
    const auto base_branch = common_parent_branch(loc_a.branch, loc_b.branch, p.morphology());
    const auto base_loc = base_branch == mnpos ? mlocation{0, 0.0} : mlocation{base_branch, 1.0};

    // compute distance to distal end of parent branch and add
    // together
    return std::abs(p.embedding().integrate_length(loc_a, base_loc)) +
           std::abs(p.embedding().integrate_length(loc_b, base_loc));
};

// Compute the distance in proximal direction. Will return nullopt if loc_prox is not between origin
// and loc_dist
std::optional<double> compute_proximal_distance(const mlocation& loc_prox,
    const mlocation& loc_dist,
    const mprovider& p) {

    // check order if on same branch
    if (loc_prox.branch == loc_dist.branch && loc_prox.pos > loc_dist.pos) return std::nullopt;

    // Special case root, for which no direction can be assumed. Always return the actual distance
    // in this case.
    if (loc_prox.pos == 0.0 && p.morphology().branch_parent(loc_prox.branch) == mnpos)
        return p.embedding().integrate_length(loc_prox, loc_dist);

    // check if loc_prox branch is in proximal direction from loc_dist
    auto b = loc_dist.branch;
    while (b > loc_prox.branch) {
        b = p.morphology().branch_parent(b);
        if (b == mnpos) return std::nullopt;
    }
    if (b != loc_prox.branch) return std::nullopt;

    return p.embedding().integrate_length(loc_prox, loc_dist);
};

enum class direction { any, proximal, distal };

// compute the minimum distance in the given direction from the given locations towards loc_eval.
// Returs nullopt if loc_eval cannot be found in the direction.
template <direction Dir>
std::optional<double> distance_from_locations(
    const std::variant<mlocation_list, mextent>& locations,
    const mlocation& loc_eval,
    const mprovider& p) {
    return std::visit(
        arb::util::overload(
            [&](const mlocation_list& arg) -> std::optional<double> {
                std::optional<double> min_dist, dist;
                for (const auto& loc: arg) {
                    if constexpr (Dir == direction::proximal) {
                        dist = compute_proximal_distance(loc_eval, loc, p);
                    }
                    else if constexpr (Dir == direction::distal) {
                        dist = compute_proximal_distance(loc, loc_eval, p);
                    }
                    else {
                        dist = compute_distance(loc, loc_eval, p);
                    }
                    if (dist)
                        min_dist = std::min(
                            min_dist.value_or(std::numeric_limits<double>::max()), dist.value());
                }
                return min_dist;
            },
            [&](const mextent& arg) -> std::optional<double> {
                std::optional<double> min_dist, dist;
                for (const auto& c: arg) {
                    if (c.branch == loc_eval.branch && c.prox_pos < loc_eval.pos &&
                        c.dist_pos > loc_eval.pos)
                        return std::nullopt;
                    if constexpr (Dir == direction::proximal) {
                        dist = compute_proximal_distance(loc_eval, {c.branch, c.prox_pos}, p);
                    }
                    else if constexpr (Dir == direction::distal) {
                        dist = compute_proximal_distance({c.branch, c.dist_pos}, loc_eval, p);
                    }
                    else {
                        dist = std::min(compute_distance({c.branch, c.dist_pos}, loc_eval, p),
                            compute_distance({c.branch, c.prox_pos}, loc_eval, p));
                    }
                    if (dist)
                        min_dist = std::min(
                            min_dist.value_or(std::numeric_limits<double>::max()), dist.value());
                }
                return min_dist;
            }),
        locations);
}

struct scalar: public iexpr_interface {
    scalar(double v): value(v) {}

    double eval(const mprovider&, const mcable&) const override { return value; }

    double value;
};

struct radius: public iexpr_interface {
    radius(double s): scale(s) {}

    double eval(const mprovider& p, const mcable& c) const override {
        auto loc_eval = mlocation{c.branch, (c.dist_pos + c.prox_pos) / 2};
        return scale * p.embedding().radius(loc_eval);
    }

    double scale;
};

struct distance: public iexpr_interface {
    distance(double s, std::variant<mlocation_list, mextent> l):
        scale(s),
        locations(std::move(l)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        auto loc_eval = mlocation{c.branch, (c.dist_pos + c.prox_pos) / 2};

        return scale *
               distance_from_locations<direction::any>(locations, loc_eval, p).value_or(0.0);
    }

    double scale;
    std::variant<mlocation_list, mextent> locations;
};

struct proximal_distance: public iexpr_interface {
    proximal_distance(double s, std::variant<mlocation_list, mextent> l):
        scale(s),
        locations(std::move(l)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        auto loc_eval = mlocation{c.branch, (c.dist_pos + c.prox_pos) / 2};

        return scale *
               distance_from_locations<direction::proximal>(locations, loc_eval, p).value_or(0.0);
    }

    double scale;
    std::variant<mlocation_list, mextent> locations;
};

struct distal_distance: public iexpr_interface {
    distal_distance(double s, std::variant<mlocation_list, mextent> l):
        scale(s),
        locations(std::move(l)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        auto loc_eval = mlocation{c.branch, (c.dist_pos + c.prox_pos) / 2};

        return scale *
               distance_from_locations<direction::distal>(locations, loc_eval, p).value_or(0.0);
    }

    double scale;
    std::variant<mlocation_list, mextent> locations;
};

struct interpolation: public iexpr_interface {
    interpolation(double prox_value,
        std::variant<mlocation_list, mextent> prox_list,
        double dist_value,
        std::variant<mlocation_list, mextent> dist_list):
        prox_v(prox_value),
        dist_v(dist_value),
        prox_l(std::move(prox_list)),
        dist_l(std::move(dist_list)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        auto loc_eval = mlocation{c.branch, (c.dist_pos + c.prox_pos) / 2};

        const auto d1 = distance_from_locations<direction::distal>(prox_l, loc_eval, p);
        if (!d1) return 0.0;

        const auto d2 = distance_from_locations<direction::proximal>(dist_l, loc_eval, p);
        if (!d2) return 0.0;

        const auto sum = d1.value() + d2.value();
        if (!sum) return (prox_v + dist_v) * 0.5;

        return prox_v * (d2.value() / sum) + dist_v * (d1.value() / sum);
    }

    double prox_v, dist_v;
    std::variant<mlocation_list, mextent> prox_l, dist_l;
};

struct add: public iexpr_interface {
    add(iexpr_ptr l, iexpr_ptr r): left(std::move(l)), right(std::move(r)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        return left->eval(p, c) + right->eval(p, c);
    }

    iexpr_ptr left;
    iexpr_ptr right;
};

struct sub: public iexpr_interface {
    sub(iexpr_ptr l, iexpr_ptr r): left(std::move(l)), right(std::move(r)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        return left->eval(p, c) - right->eval(p, c);
    }

    iexpr_ptr left;
    iexpr_ptr right;
};

struct mul: public iexpr_interface {
    mul(iexpr_ptr l, iexpr_ptr r): left(std::move(l)), right(std::move(r)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        return left->eval(p, c) * right->eval(p, c);
    }

    iexpr_ptr left;
    iexpr_ptr right;
};

struct div: public iexpr_interface {
    div(iexpr_ptr l, iexpr_ptr r): left(std::move(l)), right(std::move(r)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        return left->eval(p, c) / right->eval(p, c);
    }

    iexpr_ptr left;
    iexpr_ptr right;
};

struct exp: public iexpr_interface {
    exp(iexpr_ptr v): value(std::move(v)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        return std::exp(value->eval(p, c));
    }

    iexpr_ptr value;
};

struct step_right: public iexpr_interface {
    step_right(iexpr_ptr v): value(std::move(v)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        double x = value->eval(p, c);
        // x <  0:  0
        // x >= 0:  1
        return (x >= 0.);
    }

    iexpr_ptr value;
};

struct step_left: public iexpr_interface {
    step_left(iexpr_ptr v): value(std::move(v)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        double x = value->eval(p, c);
        // x <= 0:  0
        // x >  0:  1
        return (x > 0.);
    }

    iexpr_ptr value;
};

struct step: public iexpr_interface {
    step(iexpr_ptr v): value(std::move(v)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        double x = value->eval(p, c);
        // x <  0:  0
        // x == 0:  0.5
        // x >  0:  1
        return 0.5*((0. < x) - (x < 0.) + 1);
    }

    iexpr_ptr value;
};

struct log: public iexpr_interface {
    log(iexpr_ptr v): value(std::move(v)) {}

    double eval(const mprovider& p, const mcable& c) const override {
        return std::log(value->eval(p, c));
    }

    iexpr_ptr value;
};

}  // namespace
}  // namespace iexpr_impl

iexpr::iexpr(double value) { *this = iexpr::scalar(value); }

iexpr iexpr::scalar(double value) { return iexpr(iexpr_type::scalar, std::make_tuple(value)); }

iexpr iexpr::pi() { return iexpr::scalar(math::pi<double>); }

iexpr iexpr::distance(double scale, locset loc) {
    return iexpr(
        iexpr_type::distance, std::make_tuple(scale, std::variant<locset, region>(std::move(loc))));
}

iexpr iexpr::distance(locset loc) {
    return iexpr::distance(1.0, std::move(loc));
}

iexpr iexpr::distance(double scale, region reg) {
    return iexpr(
        iexpr_type::distance, std::make_tuple(scale, std::variant<locset, region>(std::move(reg))));
}

iexpr iexpr::distance(region reg) {
    return iexpr::distance(1.0, std::move(reg));
}

iexpr iexpr::proximal_distance(double scale, locset loc) {
    return iexpr(iexpr_type::proximal_distance,
        std::make_tuple(scale, std::variant<locset, region>(std::move(loc))));
}

iexpr iexpr::proximal_distance(locset loc) {
    return iexpr::proximal_distance(1.0, std::move(loc));
}

iexpr iexpr::proximal_distance(double scale, region reg) {
    return iexpr(iexpr_type::proximal_distance,
        std::make_tuple(scale, std::variant<locset, region>(std::move(reg))));
}

iexpr iexpr::proximal_distance(region reg) {
    return iexpr::proximal_distance(1.0, std::move(reg));
}

iexpr iexpr::distal_distance(double scale, locset loc) {
    return iexpr(iexpr_type::distal_distance,
        std::make_tuple(scale, std::variant<locset, region>(std::move(loc))));
}

iexpr iexpr::distal_distance(locset loc) {
    return iexpr::distal_distance(1.0, std::move(loc));
}

iexpr iexpr::distal_distance(double scale, region reg) {
    return iexpr(iexpr_type::distal_distance,
        std::make_tuple(scale, std::variant<locset, region>(std::move(reg))));
}

iexpr iexpr::distal_distance(region reg) {
    return iexpr::distal_distance(1.0, std::move(reg));
}

iexpr iexpr::interpolation(double prox_value,
    locset prox_list,
    double dist_value,
    locset dist_list) {
    return iexpr(iexpr_type::interpolation,
        std::make_tuple(prox_value,
            std::variant<locset, region>(std::move(prox_list)),
            dist_value,
            std::variant<locset, region>(std::move(dist_list))));
}

iexpr iexpr::interpolation(double prox_value,
    region prox_list,
    double dist_value,
    region dist_list) {
    return iexpr(iexpr_type::interpolation,
        std::make_tuple(prox_value,
            std::variant<locset, region>(std::move(prox_list)),
            dist_value,
            std::variant<locset, region>(std::move(dist_list))));
}

iexpr iexpr::radius(double scale) { return iexpr(iexpr_type::radius, std::make_tuple(scale)); }

iexpr iexpr::radius() { return iexpr::radius(1.0); }

iexpr iexpr::diameter(double scale) { return iexpr(iexpr_type::diameter, std::make_tuple(scale)); }

iexpr iexpr::diameter() { return iexpr::diameter(1.0); }

iexpr iexpr::add(iexpr left, iexpr right) {
    return iexpr(iexpr_type::add, std::make_tuple(std::move(left), std::move(right)));
}

iexpr iexpr::sub(iexpr left, iexpr right) {
    return iexpr(iexpr_type::sub, std::make_tuple(std::move(left), std::move(right)));
}

iexpr iexpr::mul(iexpr left, iexpr right) {
    return iexpr(iexpr_type::mul, std::make_tuple(std::move(left), std::move(right)));
}

iexpr iexpr::div(iexpr left, iexpr right) {
    return iexpr(iexpr_type::div, std::make_tuple(std::move(left), std::move(right)));
}

iexpr iexpr::exp(iexpr value) { return iexpr(iexpr_type::exp, std::make_tuple(std::move(value))); }

iexpr iexpr::step_right(iexpr value) { return iexpr(iexpr_type::step_right, std::make_tuple(std::move(value))); }

iexpr iexpr::step_left(iexpr value) { return iexpr(iexpr_type::step_left, std::make_tuple(std::move(value))); }

iexpr iexpr::step(iexpr value) { return iexpr(iexpr_type::step, std::make_tuple(std::move(value))); }

iexpr iexpr::log(iexpr value) { return iexpr(iexpr_type::log, std::make_tuple(std::move(value))); }

iexpr iexpr::named(std::string name) {
    return iexpr(iexpr_type::named, std::make_tuple(std::move(name)));
}

iexpr_ptr thingify(const iexpr& expr, const mprovider& m) {
    switch (expr.type()) {
    case iexpr_type::scalar:
        return iexpr_ptr(new iexpr_impl::scalar(
            std::get<0>(std::any_cast<const std::tuple<double>&>(expr.args()))));
    case iexpr_type::distance: {
        const auto& scale = std::get<0>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(expr.args()));
        const auto& var = std::get<1>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(expr.args()));

        return std::visit(
            [&](auto&& arg) {
                return iexpr_ptr(new iexpr_impl::distance(scale, thingify(arg, m)));
            },
            var);
    }
    case iexpr_type::proximal_distance: {
        const auto& scale = std::get<0>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(expr.args()));
        const auto& var = std::get<1>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(expr.args()));

        return std::visit(
            [&](auto&& arg) {
                return iexpr_ptr(new iexpr_impl::proximal_distance(scale, thingify(arg, m)));
            },
            var);
    }
    case iexpr_type::distal_distance: {
        const auto& scale = std::get<0>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(expr.args()));
        const auto& var = std::get<1>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(expr.args()));

        return std::visit(
            [&](auto&& arg) {
                return iexpr_ptr(new iexpr_impl::distal_distance(scale, thingify(arg, m)));
            },
            var);
    }
    case iexpr_type::interpolation: {
        const auto& t = std::any_cast<const std::
                tuple<double, std::variant<locset, region>, double, std::variant<locset, region>>&>(
            expr.args());
        auto prox_list = std::visit(
            [&](auto&& arg) -> std::variant<mlocation_list, mextent> { return thingify(arg, m); },
            std::get<1>(t));

        auto dist_list = std::visit(
            [&](auto&& arg) -> std::variant<mlocation_list, mextent> { return thingify(arg, m); },
            std::get<3>(t));
        return iexpr_ptr(new iexpr_impl::interpolation(
            std::get<0>(t), std::move(prox_list), std::get<2>(t), std::move(dist_list)));
    }
    case iexpr_type::radius:
        return iexpr_ptr(new iexpr_impl::radius(
            std::get<0>(std::any_cast<const std::tuple<double>&>(expr.args()))));
    case iexpr_type::diameter:
        return iexpr_ptr(new iexpr_impl::radius(
            2.0 * std::get<0>(std::any_cast<const std::tuple<double>&>(expr.args()))));
    case iexpr_type::add:
        return iexpr_ptr(new iexpr_impl::add(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr, iexpr>&>(expr.args())), m),
            thingify(std::get<1>(std::any_cast<const std::tuple<iexpr, iexpr>&>(expr.args())), m)));
    case iexpr_type::sub:
        return iexpr_ptr(new iexpr_impl::sub(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr, iexpr>&>(expr.args())), m),
            thingify(std::get<1>(std::any_cast<const std::tuple<iexpr, iexpr>&>(expr.args())), m)));
    case iexpr_type::mul:
        return iexpr_ptr(new iexpr_impl::mul(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr, iexpr>&>(expr.args())), m),
            thingify(std::get<1>(std::any_cast<const std::tuple<iexpr, iexpr>&>(expr.args())), m)));
    case iexpr_type::div:
        return iexpr_ptr(new iexpr_impl::div(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr, iexpr>&>(expr.args())), m),
            thingify(std::get<1>(std::any_cast<const std::tuple<iexpr, iexpr>&>(expr.args())), m)));
    case iexpr_type::exp:
        return iexpr_ptr(new iexpr_impl::exp(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr>&>(expr.args())), m)));
    case iexpr_type::step_right:
        return iexpr_ptr(new iexpr_impl::step_right(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr>&>(expr.args())), m)));
    case iexpr_type::step_left:
        return iexpr_ptr(new iexpr_impl::step_left(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr>&>(expr.args())), m)));
    case iexpr_type::step:
        return iexpr_ptr(new iexpr_impl::step(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr>&>(expr.args())), m)));
    case iexpr_type::log:
        return iexpr_ptr(new iexpr_impl::log(
            thingify(std::get<0>(std::any_cast<const std::tuple<iexpr>&>(expr.args())), m)));
    case iexpr_type::named:
        return m.iexpr(std::get<0>(std::any_cast<const std::tuple<std::string>&>(expr.args())));
    }

    throw std::runtime_error("thingify iexpr: Unknown iexpr type");
    return nullptr;
}

std::ostream& operator<<(std::ostream& o, const iexpr& e) {
    o << "(";

    switch (e.type()) {
    case iexpr_type::scalar: {
        o << "scalar " << std::get<0>(std::any_cast<const std::tuple<double>&>(e.args()));
        break;
    }
    case iexpr_type::distance: {
        const auto& scale = std::get<0>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(e.args()));
        const auto& var = std::get<1>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(e.args()));
        o << "distance " << scale << " ";

        std::visit([&](auto&& arg) { o << arg; }, var);
        break;
    }
    case iexpr_type::proximal_distance: {
        const auto& scale = std::get<0>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(e.args()));
        const auto& var = std::get<1>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(e.args()));
        o << "proximal-distance " << scale << " ";

        std::visit([&](auto&& arg) { o << arg; }, var);
        break;
    }
    case iexpr_type::distal_distance: {
        const auto& scale = std::get<0>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(e.args()));
        const auto& var = std::get<1>(
            std::any_cast<const std::tuple<double, std::variant<locset, region>>&>(e.args()));
        o << "distal-distance " << scale << " ";

        std::visit([&](auto&& arg) { o << arg; }, var);
        break;
    }
    case iexpr_type::interpolation: {
        using arg_type =
            std::tuple<double, std::variant<locset, region>, double, std::variant<locset, region>>;

        o << "interpolation " << std::get<0>(std::any_cast<const arg_type&>(e.args())) << " ";
        std::visit(
            [&](auto&& arg) { o << arg; }, std::get<1>(std::any_cast<const arg_type&>(e.args())));
        o << " " << std::get<2>(std::any_cast<const arg_type&>(e.args())) << " ";
        std::visit(
            [&](auto&& arg) { o << arg; }, std::get<3>(std::any_cast<const arg_type&>(e.args())));
        break;
    }
    case iexpr_type::radius: {
        o << "radius " << std::get<0>(std::any_cast<const std::tuple<double>&>(e.args()));
        break;
    }
    case iexpr_type::diameter: {
        o << "diameter " << std::get<0>(std::any_cast<const std::tuple<double>&>(e.args()));
        break;
    }
    case iexpr_type::add: {
        o << "add " << std::get<0>(std::any_cast<const std::tuple<iexpr, iexpr>&>(e.args())) << " "
          << std::get<1>(std::any_cast<const std::tuple<iexpr, iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::sub: {
        o << "sub " << std::get<0>(std::any_cast<const std::tuple<iexpr, iexpr>&>(e.args())) << " "
          << std::get<1>(std::any_cast<const std::tuple<iexpr, iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::mul: {
        o << "mul " << std::get<0>(std::any_cast<const std::tuple<iexpr, iexpr>&>(e.args())) << " "
          << std::get<1>(std::any_cast<const std::tuple<iexpr, iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::div: {
        o << "div " << std::get<0>(std::any_cast<const std::tuple<iexpr, iexpr>&>(e.args())) << " "
          << std::get<1>(std::any_cast<const std::tuple<iexpr, iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::exp: {
        o << "exp " << std::get<0>(std::any_cast<const std::tuple<iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::step_right: {
        o << "step_right " << std::get<0>(std::any_cast<const std::tuple<iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::step_left: {
        o << "step_left " << std::get<0>(std::any_cast<const std::tuple<iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::step: {
        o << "step " << std::get<0>(std::any_cast<const std::tuple<iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::log: {
        o << "log " << std::get<0>(std::any_cast<const std::tuple<iexpr>&>(e.args()));
        break;
    }
    case iexpr_type::named: {
        o << "iexpr \"" << std::get<0>(std::any_cast<const std::tuple<std::string>&>(e.args()))
          << "\"";
        break;
    }
    default: throw std::runtime_error("print iexpr: Unknown iexpr type");
    }

    o << ")";
    return o;
}

}  // namespace arb
