#include <algorithm>
#include <iostream>

#include "parameter_list.hpp"

namespace arb {

bool parameter_list::add_parameter(parameter p) {
    if (has_parameter(p.name)) {
        return false;
    }

    parameters_.push_back(std::move(p));

    return true;
}

bool parameter_list::has_parameter(std::string const& n) const {
    return find_by_name(n) != parameters_.end();
}

int parameter_list::num_parameters() const {
    return parameters_.size();
}

// returns true if parameter was succesfully updated
// returns false if parameter was not updated, i.e. if
//  - no parameter with name n exists
//  - value is not in the valid range
bool parameter_list::set(std::string const& n, parameter_list::value_type v) {
    auto p = find_by_name(n);
    if(p!=parameters_.end()) {
        if(p->is_in_range(v)) {
            p->value = v;
            return true;
        }
    }
    return false;
}

parameter& parameter_list::get(std::string const& n) {
    auto it = find_by_name(n);
    if (it==parameters_.end()) {
        throw std::domain_error(
            "parameter list does not contain parameter"
        );
    }
    return *it;
}

const parameter& parameter_list::get(std::string const& n) const {
    auto it = find_by_name(n);
    if (it==parameters_.end()) {
        throw std::domain_error(
            "parameter list does not contain parameter"
        );
    }
    return *it;
}

std::string const& parameter_list::name() const {
    return mechanism_name_;
}

std::vector<parameter> const& parameter_list::parameters() const {
    return parameters_;
}

auto parameter_list::find_by_name(std::string const& n)
    -> decltype(parameters_.begin())
{
    return
        std::find_if(
            parameters_.begin(), parameters_.end(),
            [&n](parameter const& p) {return p.name == n;}
        );
}

auto parameter_list::find_by_name(std::string const& n) const
    -> decltype(parameters_.begin())
{
    return
        std::find_if(
            parameters_.begin(), parameters_.end(),
            [&n](parameter const& p) {return p.name == n;}
        );
}

} // namespace arb

std::ostream&
operator<<(std::ostream& o, arb::parameter const& p) {
    return o
        << "parameter("
        << "name " << p.name
        << " : value " << p.value
        << " : range " << p.range
        << ")";
}

std::ostream&
operator<<(std::ostream& o, arb::parameter_list const& l) {
    o << "parameters \"" << l.name() << "\" :\n";
    for(arb::parameter const& p : l.parameters()) {
        o << " " << p << "\n";
    }
    return o;
}
