#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "error.hpp"
#include "expression.hpp"
#include "symdiff.hpp"
#include "util.hpp"
#include "visitor.hpp"

class FindIdentifierVisitor: public Visitor {
public:
    explicit FindIdentifierVisitor(const identifier_set& ids): ids_(ids) {}

    void reset() { found_ = false; }
    bool found() const { return found_; }

    void visit(Expression* e) override {}

    void visit(UnaryExpression* e) override {
        if (!found()) e->expression()->accept(this);
    }

    void visit(BinaryExpression* e) override {
        if (!found()) e->lhs()->accept(this);
        if (!found()) e->rhs()->accept(this);
    }

    void visit(CallExpression* e) override {
        for (auto& expr: e->args()) {
            if (found()) return;
            expr->accept(this);
        }
    }

    void visit(PDiffExpression* e) override {
        if (!found()) e->arg()->accept(this);
    }

    void visit(IdentifierExpression* e) override {
        if (!found()) {
            found_ |= is_in(e->spelling(), ids_);
        }
    }

    void visit(DerivativeExpression* e) override {
        if (!found()) {
            found_ |= is_in(e->spelling(), ids_);
        }
    }
    void visit(ReactionExpression* e) override {
        if (!found()) e->lhs()->accept(this);
        if (!found()) e->rhs()->accept(this);
        if (!found()) e->fwd_rate()->accept(this);
        if (!found()) e->rev_rate()->accept(this);
    }

    void visit(StoichTermExpression* e) override {
        if (!found()) e->ident()->accept(this);
    }

    void visit(StoichExpression* e) override {
        for (auto& expr: e->terms()) {
            if (found()) return;
            expr->accept(this);
        }
    }

    void visit(BlockExpression* e) override {
        for (auto& expr: e->statements()) {
            if (found()) return;
            expr->accept(this);
        }
    }

    void visit(IfExpression* e) override {
        if (!found()) e->condition()->accept(this);
        if (!found()) e->true_branch()->accept(this);
        if (!found()) e->false_branch()->accept(this);
    }

private:
    const identifier_set& ids_;
    bool found_ = false;
};

ARB_LIBMODCC_API bool involves_identifier(Expression* e, const identifier_set& ids) {
    FindIdentifierVisitor v(ids);
    e->accept(&v);
    return v.found();
}

ARB_LIBMODCC_API bool involves_identifier(Expression* e, const std::string& id) {
    identifier_set ids = {id};
    FindIdentifierVisitor v(ids);
    e->accept(&v);
    return v.found();
}

class SymPDiffVisitor: public Visitor, public error_stack {
public:
    explicit SymPDiffVisitor(std::string id): id_(std::move(id)) {}

    void reset() { result_ = nullptr; }

    // Note: moves result, forces reset.
    expression_ptr result() {
        auto r = std::move(result_);
        reset();
        return r;
    }

    void visit(Expression* e) override {
        error({"symbolic differential of improper expression", e->location()});
    }

    void visit(UnaryExpression* e) override {
        error({"symbolic differential of unrecognized unary expression", e->location()});
    }

    void visit(BinaryExpression* e) override {
        error({"symbolic differential of unrecognized binary expression", e->location()});
    }

    void visit(NegUnaryExpression* e) override {
        auto loc = e->location();
        e->expression()->accept(this);
        result_ = make_expression<NegUnaryExpression>(loc, result());
    }

    void visit(ExpUnaryExpression* e) override {
        auto loc = e->location();
        e->expression()->accept(this);
        result_ = make_expression<MulBinaryExpression>(loc, result(), e->clone());
    }

    void visit(LogUnaryExpression* e) override {
        auto loc = e->location();
        e->expression()->accept(this);
        result_ = make_expression<DivBinaryExpression>(loc, result(), e->expression()->clone());
    }

    void visit(CosUnaryExpression* e) override {
        auto loc = e->location();
        e->expression()->accept(this);
        result_ = make_expression<MulBinaryExpression>(loc,
                      make_expression<NegUnaryExpression>(loc,
                          make_expression<SinUnaryExpression>(loc, e->expression()->clone())),
                      result());
    }

    void visit(SinUnaryExpression* e) override {
        auto loc = e->location();
        e->expression()->accept(this);
        result_ = make_expression<MulBinaryExpression>(loc,
                        make_expression<CosUnaryExpression>(loc, e->expression()->clone()),
                        result());
    }

    void visit(AddBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr dlhs = std::move(result_);

        e->rhs()->accept(this);
        result_ = make_expression<AddBinaryExpression>(loc, std::move(dlhs), result());
    }

    void visit(SubBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr dlhs = std::move(result_);

        e->rhs()->accept(this);
        result_ = make_expression<SubBinaryExpression>(loc, std::move(dlhs), result());
    }

    void visit(MulBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr dlhs = std::move(result_);

        e->rhs()->accept(this);
        expression_ptr drhs = std::move(result_);

        result_ = make_expression<AddBinaryExpression>(loc,
            make_expression<MulBinaryExpression>(loc, e->lhs()->clone(), std::move(drhs)),
            make_expression<MulBinaryExpression>(loc, std::move(dlhs), e->rhs()->clone()));

    }

    void visit(DivBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr dlhs = std::move(result_);

        e->rhs()->accept(this);
        expression_ptr drhs = std::move(result_);

        result_ = make_expression<SubBinaryExpression>(loc,
            make_expression<DivBinaryExpression>(loc, std::move(dlhs), e->rhs()->clone()),
            make_expression<MulBinaryExpression>(loc,
                make_expression<DivBinaryExpression>(loc,
                    e->lhs()->clone(),
                    make_expression<MulBinaryExpression>(loc, e->rhs()->clone(), e->rhs()->clone())),
                std::move(drhs)));
    }

    void visit(PowBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr dlhs = std::move(result_);

        e->rhs()->accept(this);
        expression_ptr drhs = std::move(result_);

        result_ = make_expression<AddBinaryExpression>(loc,
            make_expression<MulBinaryExpression>(loc,
                std::move(drhs),
                make_expression<MulBinaryExpression>(loc,
                    make_expression<LogUnaryExpression>(loc, e->lhs()->clone()),
                    make_expression<PowBinaryExpression>(loc, e->lhs()->clone(), e->rhs()->clone()))),
            make_expression<MulBinaryExpression>(loc,
                e->rhs()->clone(),
                make_expression<MulBinaryExpression>(loc,
                    make_expression<PowBinaryExpression>(loc,
                        e->lhs()->clone(),
                        make_expression<SubBinaryExpression>(loc,
                            e->rhs()->clone(),
                            make_expression<IntegerExpression>(loc, 1))),
                    std::move(dlhs))));
    }

    void visit(SqrtUnaryExpression* e) override {
        auto loc = e->location();
        e->expression()->accept(this);
        // d(sqrt(f(x)))/dx = 0.5*(f(x))^(-0.5)*d(f(x))/dx
        result_ = make_expression<MulBinaryExpression>(loc,
            make_expression<MulBinaryExpression>(loc,
                make_expression<NumberExpression>(loc, 0.5),
                make_expression<PowBinaryExpression>(loc,
                    e->expression()->clone(),
                    make_expression<NumberExpression>(loc, -0.5))),
            result());
    }

    void visit(StepRightUnaryExpression* e) override {
        // ignore singularity
        auto loc = e->location();
        result_ = make_expression<IntegerExpression>(loc, 0);
    }

    void visit(StepLeftUnaryExpression* e) override {
        // ignore singularity
        auto loc = e->location();
        result_ = make_expression<IntegerExpression>(loc, 0);
    }

    void visit(StepUnaryExpression* e) override {
        // ignore singularity
        auto loc = e->location();
        result_ = make_expression<IntegerExpression>(loc, 0);
    }

    void visit(SignumUnaryExpression* e) override {
        // ignore singularity
        auto loc = e->location();
        result_ = make_expression<IntegerExpression>(loc, 0);
    }

    void visit(CallExpression* e) override {
        auto loc = e->location();
        result_ = make_expression<PDiffExpression>(loc,
                    make_expression<IdentifierExpression>(loc, id_),
                    e->clone());
    }

    void visit(PDiffExpression* e) override {
        auto loc = e->location();
        e->arg()->accept(this);
        result_ = make_expression<PDiffExpression>(loc, e->var()->clone(), result());
    }

    void visit(IdentifierExpression* e) override {
        auto loc = e->location();
        result_ = make_expression<IntegerExpression>(loc, e->spelling()==id_);
    }

    void visit(NumberExpression* e) override {
        auto loc = e->location();
        result_ = make_expression<IntegerExpression>(loc, 0);
    }

private:
    expression_ptr result_;
    std::string id_;
};

ARB_LIBMODCC_API double expr_value(Expression* e) {
    return e && e->is_number()? e->is_number()->value(): NAN;
}

class ConstantSimplifyVisitor: public Visitor {
private:
    expression_ptr result_;

    static bool is_number(Expression* e) { return e && e->is_number(); }
    static bool is_number(const expression_ptr& e) { return is_number(e.get()); }

    void as_number(Location loc, double v) {
        result_ = make_expression<NumberExpression>(loc, v);
    }

public:
    using Visitor::visit;

    ConstantSimplifyVisitor() {}

    // Note: moves result, forces reset.
    expression_ptr result() {
        auto r = std::move(result_);
        reset();
        return r;
    }

    void reset() {
        result_ = nullptr;
    }

    double value() const { return expr_value(result_); }

    bool is_number() const { return is_number(result_); }

    void visit(Expression* e) override {
        result_ = e->clone();
    }

    void visit(BlockExpression* e) override {
        auto block_ = e->clone();
        block_->is_block()->statements().clear();

        for (auto& stmt: e->statements()) {
            stmt->accept(this);
            auto simpl = result();

            // flatten any naked blocks generated by if/else simplification
            if (auto inner = simpl->is_block()) {
                for (auto& stmt: inner->statements()) {
                    block_->is_block()->statements().push_back(std::move(stmt));
                }
            }
            else {
                block_->is_block()->statements().push_back(std::move(simpl));
            }
        }
        result_ = std::move(block_);
    }

    void visit(IfExpression* e) override {
        auto loc = e->location();
        e->condition()->accept(this);
        auto cond_expr = result();
        e->true_branch()->accept(this);
        auto true_expr = result();
        expression_ptr false_expr;
        if (e->false_branch()) {
            e->false_branch()->accept(this);
            false_expr = result()->clone();
        }

        if (!is_number(cond_expr)) {
            result_ = make_expression<IfExpression>(loc,
                        std::move(cond_expr), std::move(true_expr), std::move(false_expr));
        }
        else if (expr_value(cond_expr)) {
            result_ = std::move(true_expr);
        }
        else {
            result_ = std::move(false_expr);
        }
    }

    // TODO: procedure, function expressions

    void visit(UnaryExpression* e) override {
        e->expression()->accept(this);

        if (is_number()) {
            auto loc = e->location();
            auto val = value();

            switch (e->op()) {
            case tok::minus:
                as_number(loc, -val);
                return;
            case tok::exp:
                as_number(loc, std::exp(val));
                return;
            case tok::sin:
                as_number(loc, std::sin(val));
                return;
            case tok::cos:
                as_number(loc, std::cos(val));
                return;
            case tok::log:
                as_number(loc, std::log(val));
                return;
            case tok::sqrt:
                as_number(loc, std::sqrt(val));
                return;
            case tok::step_right:
                as_number(loc, (val >= 0.));
                return;
            case tok::step_left:
                as_number(loc, (val > 0.));
                return;
            case tok::step:
                as_number(loc, 0.5*((0. < val) - (val < 0.) + 1));
                return;
            case tok::signum:
                as_number(loc, (0. < val) - (val < 0.));
                return;
            default: ; // treat opaquely as below
            }
        }

        expression_ptr arg = result();
        result_ = e->clone();
        result_->is_unary()->replace_expression(std::move(arg));
    }

    void visit(BinaryExpression* e) override {
        result_ = e->clone();
    }

    void visit(AssignmentExpression* e) override {
        auto loc = e->location();
        e->rhs()->accept(this);
        result_ = make_expression<AssignmentExpression>(loc, e->lhs()->clone(), result());
    }

    void visit(MulBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr lhs = result();
        e->rhs()->accept(this);
        expression_ptr rhs = result();

        if (is_number(lhs) && is_number(rhs)) {
            as_number(loc, expr_value(lhs)*expr_value(rhs));
        }
        else if (expr_value(lhs)==0 || expr_value(rhs)==0) {
            as_number(loc, 0);
        }
        else if (expr_value(lhs)==1) {
            result_ = std::move(rhs);
        }
        else if (expr_value(rhs)==1) {
            result_ = std::move(lhs);
        }
        else {
            result_ = make_expression<MulBinaryExpression>(loc, std::move(lhs), std::move(rhs));
        }
    }

    void visit(DivBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr lhs = result();
        e->rhs()->accept(this);
        expression_ptr rhs = result();

        if (is_number(lhs) && is_number(rhs)) {
            as_number(loc, expr_value(lhs)/expr_value(rhs));
        }
        else if (expr_value(lhs)==0) {
            as_number(loc, 0);
        }
        else if (expr_value(rhs)==1) {
            result_ = e->lhs()->clone();
        }
        else if (is_number(rhs)) {
            result_ = make_expression<MulBinaryExpression>(loc, std::move(lhs), make_expression<NumberExpression>(loc, 1.0/expr_value(rhs)));
        }
        else {
            result_ = make_expression<DivBinaryExpression>(loc, std::move(lhs), std::move(rhs));
        }
    }

    void visit(AddBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr lhs = result();
        e->rhs()->accept(this);
        expression_ptr rhs = result();

        if (is_number(lhs) && is_number(rhs)) {
            as_number(loc, expr_value(lhs)+expr_value(rhs));
        }
        else if (expr_value(lhs)==0) {
            result_ = std::move(rhs);
        }
        else if (expr_value(rhs)==0) {
            result_ = std::move(lhs);
        }
        else {
            result_ = make_expression<AddBinaryExpression>(loc, std::move(lhs), std::move(rhs));
        }
    }

    void visit(SubBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr lhs = result();
        e->rhs()->accept(this);
        expression_ptr rhs = result();

        if (is_number(lhs) && is_number(rhs)) {
            as_number(loc, expr_value(lhs)-expr_value(rhs));
        }
        else if (expr_value(lhs)==0) {
            make_expression<NegUnaryExpression>(loc, std::move(rhs))->accept(this);
        }
        else if (expr_value(rhs)==0) {
            result_ = std::move(lhs);
        }
        else {
            result_ = make_expression<SubBinaryExpression>(loc, std::move(lhs), std::move(rhs));
        }
    }

    void visit(PowBinaryExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr lhs = result();
        e->rhs()->accept(this);
        expression_ptr rhs = result();

        if (is_number(lhs) && is_number(rhs)) {
            as_number(loc, std::pow(expr_value(lhs),expr_value(rhs)));
        }
        else if (expr_value(lhs)==0) {
            as_number(loc, 0);
        }
        else if (expr_value(rhs)==0 || expr_value(lhs)==1) {
            as_number(loc, 1);
        }
        else if (expr_value(rhs)==1) {
            result_ = std::move(lhs);
        }
        else {
            result_ = make_expression<PowBinaryExpression>(loc, std::move(lhs), std::move(rhs));
        }
    }

    void visit(ConditionalExpression* e) override {
        auto loc = e->location();
        e->lhs()->accept(this);
        expression_ptr lhs = result();
        e->rhs()->accept(this);
        expression_ptr rhs = result();

        if (is_number(lhs) && is_number(rhs)) {
            auto lval = expr_value(lhs);
            auto rval = expr_value(rhs);
            switch (e->op()) {
            case tok::equality:
                as_number(loc, lval==rval);
                return;
            case tok::ne:
                as_number(loc, lval!=rval);
                return;
            case tok::lt:
                as_number(loc, lval<rval);
                return;
            case tok::gt:
                as_number(loc, lval>rval);
                return;
            case tok::lte:
                as_number(loc, lval<=rval);
                return;
            case tok::gte:
                as_number(loc, lval>=rval);
            case tok::land:
                as_number(loc, lval&&rval);
            case tok::lor:
                as_number(loc, lval||rval);
                    return;
            default: ;
                // unrecognized, fall through to non-numeric case below
            }
        }
        if (!is_number(lhs) || !is_number(rhs)) {
            result_ = make_expression<ConditionalExpression>(loc, e->op(), std::move(lhs), std::move(rhs));
        }
    }
};

ARB_LIBMODCC_API expression_ptr constant_simplify(Expression* e) {
    ConstantSimplifyVisitor csimp_visitor;
    e->accept(&csimp_visitor);
    return csimp_visitor.result();
}


ARB_LIBMODCC_API expression_ptr symbolic_pdiff(Expression* e, const std::string& id) {
    if (!involves_identifier(e, id)) {
        return make_expression<NumberExpression>(e->location(), 0);
    }

    SymPDiffVisitor pdiff_visitor(id);
    e->accept(&pdiff_visitor);

    if (pdiff_visitor.has_error()) {
        std::string errors, sep = "";

        for (const auto& error: pdiff_visitor.errors()) {
            errors += sep + error.message;
            sep = "\n";
        }
        auto res = std::make_unique<ErrorExpression>(e->location());
        res->error(errors);
        return res;
    }

    return constant_simplify(pdiff_visitor.result());
}

// Substitute all occurances of an identifier within a unary, binary, call
// or (trivially) number expression with a copy of the provided substitute
// expression.

class SubstituteVisitor: public Visitor {
public:
    explicit SubstituteVisitor(const substitute_map& sub):
        sub_(sub) {}

    expression_ptr result() {
        auto r = std::move(result_);
        reset();
        return r;
    }

    void reset() {
        result_ = nullptr;
    }

    void visit(Expression* e) override {
        throw compiler_exception("substitution attempt on improper expression", e->location());
    }

    void visit(NumberExpression* e) override {
        result_ = e->clone();
    }

    void visit(IdentifierExpression* e) override {
        result_ = is_in(e->spelling(), sub_)? sub_.at(e->spelling())->clone(): e->clone();
    }

    void visit(UnaryExpression* e) override {
        e->expression()->accept(this);
        auto arg = result();

        result_ = e->clone();
        result_->is_unary()->replace_expression(std::move(arg));
    }

    void visit(BinaryExpression* e) override {
        e->lhs()->accept(this);
        auto lhs = result();
        e->rhs()->accept(this);
        auto rhs = result();

        result_ = e->clone();
        result_->is_binary()->replace_lhs(std::move(lhs));
        result_->is_binary()->replace_rhs(std::move(rhs));
    }

    void visit(CallExpression* e) override {
        auto newexpr = e->clone();
        for (auto& arg: newexpr->is_call()->args()) {
            arg->accept(this);
            arg = result();
        }
        result_ = std::move(newexpr);
    }

    void visit(PDiffExpression* e) override {
        // Doing the correct thing when the derivative variable is the
        // substitution variable would require another 'opaque' expression,
        // e.g. `SubstitutionExpression`, but we're not about to do that yet,
        // so throw an exception instead, i.e. Don't Do That.
        if (is_in(e->var()->is_identifier()->spelling(), sub_)) {
            throw compiler_exception("attempt to substitute value for derivative variable", e->location());
        }

        e->arg()->accept(this);
        result_ = make_expression<PDiffExpression>(e->location(), e->var()->clone(), result());
    }

private:
    expression_ptr result_;
    const substitute_map& sub_;
};

ARB_LIBMODCC_API expression_ptr substitute(Expression* e, const std::string& id, Expression* sub) {
    substitute_map subs;
    subs[id] = sub->clone();
    SubstituteVisitor sub_visitor(subs);
    e->accept(&sub_visitor);
    return sub_visitor.result();
}

ARB_LIBMODCC_API expression_ptr substitute(Expression* e, const substitute_map& sub) {
    SubstituteVisitor sub_visitor(sub);
    e->accept(&sub_visitor);
    return sub_visitor.result();
}

ARB_LIBMODCC_API linear_test_result linear_test(Expression* e, const std::vector<std::string>& vars) {
    linear_test_result result;
    auto loc = e->location();
    auto zero = [loc]() { return make_expression<IntegerExpression>(loc, 0); };

    result.constant = e->clone();
    for (const auto& id: vars) {
        auto coef = symbolic_pdiff(e, id);
        if (coef->has_error()) {
            auto res = linear_test_result{};
            res.error({coef->error_message(), loc});
            return res;
        }
        if (!coef) return linear_test_result{};
        if (!is_zero(coef)){
            result.coef[id] = std::move(coef);
            result.is_constant = false;
        }
        result.constant = substitute(result.constant, id, zero());
    }

    ConstantSimplifyVisitor csimp_visitor;
    result.constant->accept(&csimp_visitor);
    result.constant = csimp_visitor.result();
    if (result.constant.get() == nullptr) throw compiler_exception{"Linear test: simplification of the constant term failed.", loc};

    // linearity test: take second order derivatives, test against zero.
    result.is_linear = true;
    for (unsigned i = 0; i<vars.size(); ++i) {
        auto v1 = vars[i];
        if (!is_in(v1, result.coef)) continue;

        for (unsigned j = i; j<vars.size(); ++j) {
            auto v2 = vars[j];
            auto coef = symbolic_pdiff(result.coef[v1].get(), v2);
            if (!coef || !is_zero(coef.get())) {
                result.is_linear = false;
                goto done;
            }
        }
    }
done:

    if (result.is_linear) {
        result.is_homogeneous = is_zero(result.constant);
    }

    return result;
}
