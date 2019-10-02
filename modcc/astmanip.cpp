#include <string>

#include "astmanip.hpp"
#include "expression.hpp"
#include "location.hpp"
#include "scope.hpp"

static std::string unique_local_name(scope_ptr scope, std::string const& prefix) {
    for (int i = 0; ; ++i) {
        std::string name = prefix + std::to_string(i) + "_";
        if (!scope->find(name)) return name;
    }
}

local_assignment make_unique_local_assign(scope_ptr scope, Expression* e, std::string const& prefix) {
    Location loc = e->location();
    std::string name = unique_local_name(scope, prefix);

    auto local = make_expression<LocalDeclaration>(loc, name);
    local->semantic(scope);

    auto id = make_expression<IdentifierExpression>(loc, name);
    id->semantic(scope);

    auto ass = binary_expression(e->location(), tok::eq, id->clone(), e->clone());
    ass->semantic(scope);

    return { std::move(local), std::move(ass), std::move(id), scope };
}

local_declaration make_unique_local_decl(scope_ptr scope, Location loc, std::string const& prefix) {
    std::string name = unique_local_name(scope, prefix);

    auto local = make_expression<LocalDeclaration>(loc, name);
    local->semantic(scope);

    auto id = make_expression<IdentifierExpression>(loc, name);
    id->semantic(scope);

    return { std::move(local), std::move(id), scope };
}
