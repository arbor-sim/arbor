#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "io/pprintf.hpp"


// Scope is templated to avoid circular compilation issues.
// When performing semantic analysis of expressions via traversal of the AST
// each node in the AST has a reference to a Scope. This leads to circular
// dependencies, where Symbol nodes refer to Scopes which contain Symbols.
// Using a template means that we can defer Scope definition until after
// the Symbol type defined in expression.h has been defined.
template <typename Symbol>
class Scope {
public:
    using symbol_type = Symbol;
    using symbol_ptr  = std::unique_ptr<Symbol>;
    using symbol_map  = std::unordered_map<std::string, symbol_ptr>;

    Scope(symbol_map& s);
    ~Scope() {};
    symbol_type* add_local_symbol(std::string const& name, symbol_ptr s);
    symbol_type* find(std::string const& name) const;
    symbol_type* find_local(std::string const& name) const;
    symbol_type* find_global(std::string const& name) const;
    std::string to_string() const;

    symbol_map& locals();
    symbol_map* globals();

    bool in_api_context() const {
        return api_context_;
    }

    void in_api_context(bool flag) {
        api_context_ = flag;
    }

private:
    symbol_map* global_symbols_=nullptr;
    symbol_map  local_symbols_;
    bool api_context_ = false;
};

template<typename Symbol>
Scope<Symbol>::Scope(symbol_map &s)
    : global_symbols_(&s)
{}

template<typename Symbol>
Symbol*
Scope<Symbol>::add_local_symbol( std::string const& name,
                         typename Scope<Symbol>::symbol_ptr s)
{
    // check to see if the symbol already exists
    if( local_symbols_.find(name) != local_symbols_.end() ) {
        return nullptr;
    }

    // add symbol to list
    local_symbols_[name] = std::move(s);

    return local_symbols_[name].get();
}

template<typename Symbol>
Symbol*
Scope<Symbol>::find(std::string const& name) const {
    auto local = find_local(name);
    return local ? local : find_global(name);
}

template<typename Symbol>
Symbol*
Scope<Symbol>::find_local(std::string const& name) const {
    // search in local symbols
    auto local = local_symbols_.find(name);

    if(local != local_symbols_.end()) {
        return local->second.get();
    }

    return nullptr;
}

template<typename Symbol>
Symbol*
Scope<Symbol>::find_global(std::string const& name) const {
    // search in global symbols
    if( global_symbols_ ) {
        auto global = global_symbols_->find(name);

        if(global != global_symbols_->end()) {
            return global->second.get();
        }
    }

    return nullptr;
}

template<typename Symbol>
std::string
Scope<Symbol>::to_string() const {
    std::string s;
    char buffer[16];

    s += blue("Scope") + "\n";
    s += blue("  global :\n");
    for(auto& sym : *global_symbols_) {
        snprintf(buffer, 16, "%-15s", sym.first.c_str());
        s += "    " + yellow(buffer);
    }
    s += "\n";
    s += blue("  local  :\n");
    for(auto& sym : local_symbols_) {
        snprintf(buffer, 16, "%-15s", sym.first.c_str());
        s += "    " + yellow(buffer);
    }

    return s;
}

template<typename Symbol>
typename Scope<Symbol>::symbol_map&
Scope<Symbol>::locals() {
    return local_symbols_;
}

template<typename Symbol>
typename Scope<Symbol>::symbol_map*
Scope<Symbol>::globals() {
    return global_symbols_;
}

