#include <regex>
#include <string>
#include <unordered_set>

#include "expression.hpp"
#include "module.hpp"
#include "printerutil.hpp"

std::vector<std::string> namespace_components(const std::string& ns) {
    static std::regex ns_regex("([^:]+)(?:::|$)");

    std::vector<std::string> components;
    auto i = std::sregex_iterator(ns.begin(), ns.end(), ns_regex);
    while (i!=std::sregex_iterator()) {
        components.push_back(i++->str(1));
    }

    return components;
}

std::vector<LocalVariable*> indexed_locals(scope_ptr scope) {
    std::vector<LocalVariable*> vars;
    for (auto& entry: scope->locals()) {
        LocalVariable* local = entry.second->is_local_variable();
        if (local && local->is_indexed()) {
            vars.push_back(local);
        }
    }
    return vars;
}

std::vector<LocalVariable*> pure_locals(scope_ptr scope) {
    std::vector<LocalVariable*> vars;
    for (auto& entry: scope->locals()) {
        LocalVariable* local = entry.second->is_local_variable();
        if (local && !local->is_arg() && !local->is_indexed()) {
            vars.push_back(local);
        }
    }
    return vars;
}

std::vector<ProcedureExpression*> normal_procedures(const Module& m) {
    std::vector<ProcedureExpression*> procs;

    for (auto& sym: m.symbols()) {
        auto proc = sym.second->is_procedure();
        if (proc && proc->kind()==procedureKind::normal && !proc->is_api_method() && !proc->is_net_receive()) {
            procs.push_back(proc);
        }
    }

    return procs;
}

public_variable_ids_t public_variable_ids(const Module& m) {
    public_variable_ids_t ids;
    ids.state_ids = m.state_block().state_variables;

    std::unordered_set<std::string> range_varnames;
    for (const auto& sym: m.symbols()) {
        if (auto var = sym.second->is_variable()) {
            if (var->is_range()) {
                range_varnames.insert(var->name());
            }
        }
    }

    for (const Id& id: m.parameter_block().parameters) {
        if (range_varnames.count(id.token.spelling)) {
            ids.range_parameter_ids.push_back(id);
        }
        else {
            ids.global_parameter_ids.push_back(id);
        }
    }

    return ids;
}

module_variables_t local_module_variables(const Module& m) {
    module_variables_t mv;

    for (auto& sym: m.symbols()) {
        auto v = sym.second->is_variable();
        if (v && v->linkage()==linkageKind::local) {
            (v->is_range()? mv.arrays: mv.scalars).push_back(v);
        }
    }

    return mv;
}

APIMethod* find_api_method(const Module& m, const char* which) {
    auto it = m.symbols().find(which);
    return  it==m.symbols().end()? nullptr: it->second->is_api_method();
}

NetReceiveExpression* find_net_receive(const Module& m) {
    auto it = m.symbols().find("net_receive");
    return it==m.symbols().end()? nullptr: it->second->is_net_receive();
}
