#pragma once

#include <any>
#include <string>

#include <arborio/label_parse.hpp>

#include <arbor/morph/label_dict.hpp>

#include "strprintf.hpp"

namespace pyarb {

struct iexpr_proxy {
    ::arb::iexpr iexpr;
    std::string label;
};

struct label_dict_proxy {
    using str_map = std::unordered_map<std::string, std::string>;
    arb::label_dict dict;
    str_map cache;
    std::vector<std::string> locsets;
    std::vector<std::string> regions;
    std::vector<std::string> iexpressions;

    label_dict_proxy() = default;

    label_dict_proxy(const str_map& in) {
        for (auto& i: in) {
            set(i.first, i.second);
        }
    }

    label_dict_proxy& add_swc_tags() {
        set("soma", "(tag 1)");
        set("axon", "(tag 2)");
        set("dend", "(tag 3)");
        set("apic", "(tag 4)");
        return *this;
    }

    label_dict_proxy(const arb::label_dict& label_dict): dict(label_dict) {
        update_cache();
    }

    label_dict_proxy(const label_dict_proxy&) = default;

    std::size_t size() const  {
        return locsets.size() + regions.size() + iexpressions.size();
    }

    void import(const label_dict_proxy& other, std::string prefix = "") {
        dict.import(other.dict, prefix);

        clear_cache();
        update_cache();
    }

    void set(const std::string& name, const std::string& desc) {
        using namespace std::string_literals;
        // The following code takes an input name and a region or locset or iexpr
        // description, e.g.:
        //      name='reg', desc='(tag 4)'
        //      name='loc', desc='(terminal)'
        //      name='foo', desc='(join (tag 2) (tag 3))'
        // Then it parses the description, and tests whether the description
        // is a region or locset or iexpr, and updates the label dictionary appropriately.
        // Errors occur when:
        //  * a region is described with a name that matches an existing locset or iexpr
        //    (and vice versa.)
        //  * the description is not well formed, e.g. it contains a syntax error.
        //  * the description is well-formed, but describes neither a region or locset or iexpr.
        try {
            // Evaluate the s-expression to build a region/locset/iexpr.
            auto result = arborio::parse_label_expression(desc);
            if (!result) { // an error parsing / evaluating description.
                throw result.error();
            }
            else if (result->type()==typeid(arb::region)) { // describes a region.
                dict.set(name, std::move(std::any_cast<arb::region&>(*result)));
                auto it = std::lower_bound(regions.begin(), regions.end(), name);
                if (it==regions.end() || *it!=name) regions.insert(it, name);
            }
            else if (result->type()==typeid(arb::locset)) { // describes a locset.
                dict.set(name, std::move(std::any_cast<arb::locset&>(*result)));
                auto it = std::lower_bound(locsets.begin(), locsets.end(), name);
                if (it==locsets.end() || *it!=name) locsets.insert(it, name);
            }
            else if (result->type()==typeid(arb::iexpr)) { // describes a iexpr.
                dict.set(name, std::move(std::any_cast<arb::iexpr&>(*result)));
                auto it = std::lower_bound(iexpressions.begin(), iexpressions.end(), name);
                if (it==iexpressions.end() || *it!=name) iexpressions.insert(it, name);
            }
            else {
                // Successfully parsed an expression that is neither region nor locset.
                throw util::pprintf("The definition of '{} = {}' does not define a valid region or locset.", name, desc);
            }
            // The entry was added succesfully: store it in the cache.
            cache[name] = desc;
        }
        catch (std::string msg) {
            const char* base = "\nError adding the label '{}' = '{}'\n{}\n";

            throw std::runtime_error(util::pprintf(base, name, desc, msg));
        }
        // Exceptions are thrown in parse or eval if an unexpected error occured.
        catch (std::exception& e) {
            const char* msg =
                "\n----- internal error -------------------------------------------"
                "\nError parsing the label: '{}' = '{}'"
                "\n"
                "\n{}"
                "\n"
                "\nPlease file a bug report with this full error message at:"
                "\n    github.com/arbor-sim/arbor/issues"
                "\n----------------------------------------------------------------";
            throw arb::arbor_internal_error(util::pprintf(msg, name, desc, e.what()));
        }
    }

    std::string to_string() const {
        std::string s;
        s += "(label_dict";
        for (auto& x: dict.regions()) {
            s += util::pprintf(" (region  \"{}\" {})", x.first, x.second);
        }
        for (auto& x: dict.locsets()) {
            s += util::pprintf(" (locset \"{}\" {})", x.first, x.second);
        }
        for (auto& x: dict.iexpressions()) {
            s += util::pprintf(" (iexpr \"{}\" {})", x.first, x.second);
        }
        s += ")";
        return s;
    }

    bool contains(const std::string& name) const {
        return cache.find(name) != cache.end();
    }

    std::optional<std::string> getitem(const std::string& name) const {
        if (auto kv = cache.find(name); kv != cache.end()) {
            return kv->second;
        }
        return {};
    }

    private:

    void clear_cache() {
        regions.clear();
        locsets.clear();
        iexpressions.clear();
        cache.clear();
    }

    void update_cache() {
        for (const auto& [lab, reg]: dict.regions()) {
            if (!cache.count(lab)) {
                std::stringstream s;
                s << reg;
                regions.push_back(lab);
                cache[lab] = s.str();
            }
        }
        for (const auto& [lab, ls]: dict.locsets()) {
            if (!cache.count(lab)) {
                std::stringstream s;
                s << ls;
                locsets.push_back(lab);
                cache[lab] = s.str();
            }
        }
        for (const auto& [lab, ls]: dict.iexpressions()) {
            if (!cache.count(lab)) {
                std::stringstream s;
                s << ls;
                iexpressions.push_back(lab);
                cache[lab] = s.str();
            }
        }
        // Sort the region and locset names
        std::sort(regions.begin(), regions.end());
        std::sort(locsets.begin(), locsets.end());
        std::sort(iexpressions.begin(), iexpressions.end());
    }
};
}
