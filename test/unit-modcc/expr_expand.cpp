#include <stdexcept>
#include <sstream>

#include "expression.hpp"
#include "io/pprintf.hpp"
#include "token.hpp"

#include "alg_collect.hpp"
#include "expr_expand.hpp"

alg::prodsum expand_expression(Expression* e, const id_prodsum_map& exmap) {
    using namespace alg;

    if (const auto& n = e->is_number()) {
        return prodsum(n->value());
    }
    else if (const auto& c = e->is_function_call()) {
        std::stringstream rep(c->name());
        rep << '(';
        bool first = true;
        for (const auto& arg: c->args()) {
            if (!first) rep << ',';
            rep << expand_expression(arg.get(), exmap);
            first = false;
        }
        rep << ')';
        return prodterm(rep.str());
    }
    else if (const auto& i = e->is_identifier()) {
        std::string k = i->spelling();
        auto x = exmap.find(k);
        return x!=exmap.end()? x->second: prodterm(k);
    }
    else if (const auto& b = e->is_binary()) {
        prodsum lhs = expand_expression(b->lhs(), exmap);
        prodsum rhs = expand_expression(b->rhs(), exmap);

        switch (b->op()) {
        case tok::plus:
            return lhs+rhs;
        case tok::minus:
            return lhs-rhs;
        case tok::times:
            return lhs*rhs;
        case tok::divide:
            return lhs/rhs;
        case tok::pow:
            if (!rhs.is_scalar()) {
                // make an opaque term for this case (i.e. too hard to simplify)
                return prodterm(pprintf("(%)^(%)", lhs, rhs));
            }
            else return lhs.pow(rhs.first_coeff());
        default:
            throw std::runtime_error("unrecognized binop");
        }
    }
    else if (const auto& u = e->is_unary()) {
        prodsum inner = expand_expression(u->expression(), exmap);
        switch (u->op()) {
        case tok::minus:
            return -inner;
        case tok::exp:
            return prodterm(pprintf("exp(%)", inner));
        case tok::log:
            return prodterm(pprintf("log(%)", inner));
        case tok::sin:
            return prodterm(pprintf("sin(%)", inner));
        case tok::cos:
            return prodterm(pprintf("cos(%)", inner));
        default:
            throw std::runtime_error("unrecognized unaryop");
        }
    }
    else {
        throw std::runtime_error("unexpected expression type");
    }
}
