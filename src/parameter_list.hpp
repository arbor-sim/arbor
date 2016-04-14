#pragma once

#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nest {
namespace mc {

    template <typename T>
    struct value_range {
        using value_type = T;

        value_range()
        :   min(std::numeric_limits<value_type>::quiet_NaN()),
            max(std::numeric_limits<value_type>::quiet_NaN())
        { }

        value_range(value_type const& left, value_type const& right)
        :   min(left),
            max(right)
        {
            if(left>right) {
                throw std::out_of_range(
                    "parameter range must have left <= right"
                );
            }
        }

        bool has_lower_bound() const
        {
            return min==min;
        }

        bool has_upper_bound() const
        {
            return max==max;
        }

        bool is_in_range(value_type const& v) const
        {
            if(has_lower_bound()) {
                if(v<min) return false;
            }
            if(has_upper_bound()) {
                if(v>max) return false;
            }
            return true;
        }

        value_type min;
        value_type max;
    };

    template <typename T>
    std::ostream& operator<<(std::ostream& o, value_range<T> const& r)
    {
        return
            o << "[ "
              << (r.has_lower_bound() ? std::to_string(r.min) : "-inf") << ", "
              << (r.has_upper_bound() ? std::to_string(r.max) : "inf") << "]";
    }

    struct parameter {
        using value_type = double;
        using range_type = value_range<value_type>;

        parameter() = delete;

        parameter(std::string n, value_type v, range_type r=range_type())
        :   name(std::move(n)),
            value(v),
            range(r)
        {
            if(!is_in_range(v)) {
                throw std::out_of_range(
                    "parameter value is out of permitted value range"
                );
            }
        }

        bool is_in_range(value_type v) const
        {
            return range.is_in_range(v);
        }

        std::string name;
        value_type value;
        range_type range;
    };

    std::ostream& operator<<(std::ostream& o, parameter const& p);

    // Use a dumb container class for now
    // might have to use a more sophisticated interface in the future if need be
    class parameter_list {
        public :

        using value_type = double;

        parameter_list(std::string n)
        : mechanism_name_(std::move(n))
        { }

        bool has_parameter(std::string const& n) const;
        bool add_parameter(parameter p);

        // returns true if parameter was succesfully updated
        // returns false if parameter was not updated, i.e. if
        //  - no parameter with name n exists
        //  - value is not in the valid range
        bool set(std::string const& n, value_type v);
        parameter& get(std::string const& n);

        std::string const& name() const;

        std::vector<parameter> const& parameters() const;

        int num_parameters() const;

        private:

        std::vector<parameter> parameters_;
        std::string mechanism_name_;

        // need two versions for const and non-const applications
        auto find_by_name(std::string const& n)
            -> decltype(parameters_.begin());

        auto find_by_name(std::string const& n) const
            -> decltype(parameters_.begin());

    };

    std::ostream& operator<<(std::ostream& o, parameter_list const& l);

    ///////////////////////////////////////////////////////////////////////////
    //  predefined parameter sets
    ///////////////////////////////////////////////////////////////////////////

    /// default set of parameters for the cell membrane that are added to every
    /// segment when it is created.
    class membrane_parameters
    : public parameter_list
    {
        public:

        using base = parameter_list;

        using base::value_type;

        using base::set;
        using base::get;
        using base::parameters;
        using base::has_parameter;

        membrane_parameters()
        : base("membrane")
        {
            base::add_parameter({"r_L",   0.01, {0., 1e9}}); // typically 10 nF/mm^2 == 0.01 F/m2
            base::add_parameter({"c_m", 180.00, {0., 1e9}}); // Ohm.cm
        }
    };

    /// parameters for the classic Hodgkin & Huxley model (1952)
    class hh_parameters
    : public parameter_list
    {
        public:

        using base = parameter_list;

        using base::value_type;

        using base::set;
        using base::get;
        using base::parameters;
        using base::has_parameter;

        hh_parameters()
        : base("hh")
        {
            base::add_parameter({"gnabar", 0.12,  {0., 1e9}});
            base::add_parameter({"gkbar",  0.036, {0., 1e9}});
            base::add_parameter({"gl",     0.0003,{0., 1e9}});
            base::add_parameter({"el",     -54.3});
        }
    };

} // namespace mc
} // namespace nest

