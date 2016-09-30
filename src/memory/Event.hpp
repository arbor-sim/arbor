#pragma once

#include <utility>

#include "definitions.hpp"

namespace memory {

enum EventStatus {kEventBusy, kEventReady};

namespace util {
/*
template<class T1, class T2>
struct pair
{
    typedef T1 first_type;
    typedef T2 second_type;

    T1 first;
    T2 second;

    pair()
    :   first(),
        second()
    {}

    pair(const T1& a, const T2& b)
    :   first(a),
        second(b)
    {}

    template<class U1, class U2>
    pair(const pair<U1, U2>& p)
    :   first(p.first),
        second(p.second)
    {}

    template<class U1, class U2>
    pair(U1&& x, U2&& y)
    :    first(std::forward<U1>(x)),
         second(std::forward<U2>(y))
    {}

    pair(pair&& p)
    :    first(std::move(p.first)),
         second(std::move(p.second))
    {}
};

template<class T1, class T2>
inline pair<T1, T2>
make_pair(T1 x, T2 y) {
    return pair<T1, T2>(x, y);
}

template <typename First, typename Second>
struct type_printer<pair<First, Second>>{
    static std::string print() {
        std::stringstream str;
        str << "util::pair<" << type_printer<First>::print()
            << ", " << type_printer<Second>::print() << ">";
        return str.str();
    }
};
*/
} // namespace util

// empty event that can be used for synchronous events, or for events that are
// guaranteed to have completed when the event will be queried
class SynchEvent {
public:
    SynchEvent() = default;

    // pause execution in calling thread until event is finished
    // this returns instantly for a synchronous event
    void wait() {};

    EventStatus query() {
        return kEventReady;
    }
};

// abstract base class for an asynchronous event
class AsynchEvent {
public:
    // pause execution in calling thread until event is finished
    // this returns instantly for a synchronous event
    virtual void wait() = 0;

    virtual EventStatus query() = 0;
};

// abstract base class for an asynchronous event
class CUDAEvent
: public AsynchEvent {
public:
    CUDAEvent() = default;

    virtual void
    wait() override {}

    virtual EventStatus
    query() override {
        return kEventReady;
    }
};

namespace util {
    template <>
    struct pretty_printer<SynchEvent>{
        static std::string print(const SynchEvent&) {
            return std::string("SynchEvent()");
        }
    };

    template <>
    struct pretty_printer<CUDAEvent>{
        static std::string print(const CUDAEvent& event) {
            return std::string("CUDAEvent()");
        }
    };

    template <>
    struct type_printer<SynchEvent>{
        static std::string print() {
            return std::string("SynchEvent");
        }
    };

    template <>
    struct type_printer<CUDAEvent>{
        static std::string print() {
            return std::string("CUDAEvent");
        }
    };
} // namespace util

} // namespace events
