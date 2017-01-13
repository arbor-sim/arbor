#pragma once

#include "lib/vector/include/Vector.h"

/*
    Abstract base class for all mechanisms

    This works well for the standard interface that is exported by all mechanisms,
    i.e. nrn_jacobian(), nrn_current(), etc. The overhead of using virtual dispatch
    for such functions is negligable compared to the cost of the operations themselves.
    However, the friction between compile time and run time dispatch has to be considered
    carefully:
        - we want to dispatch on template parameters like vector width, target
          hardware etc. Maybe these could be template parameters to the base
          class?
        - how to expose mechanism functionality, e.g. for computing a mechanism-
          specific quantity for visualization?
*/

class Mechanism {
public:
    // typedefs for storage
    using value_type = double;
    using array = memory::HostVector<value_type>;
    using view_type   = memory::HostView<value_type>;

    Mechanism(std::string const& name)
    :   name_(name)
    {}

    //virtual void state()   = 0;
    //virtual void jacobi()  = 0;
    virtual void current() = 0;
    virtual void init()    = 0;

    std::string const& name() {
        return name_;
    }

protected:
    std::string name_;
};

