#pragma once

#include <cstddef>
#include <sstream>
#include <typeinfo>

namespace arb {
namespace memory {

namespace types {
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
} // namespace types

namespace util {

    // forward declare type printer
    template <typename T> struct type_printer;

    template <typename T>
    struct pretty_printer{
        static std::string print(const T& /*val*/) {
            return "T()";
        }
    };

    template <>
    struct pretty_printer<float>{
        static std::string print(const float& val) {
            std::stringstream str;
            str << "float(" << val << ")";
            return str.str();
        }
    };

    template <>
    struct pretty_printer<double>{
        static std::string print(const double& val) {
            std::stringstream str;
            str << "double(" << val << ")";
            return str.str();
        }
    };

    template <>
    struct pretty_printer<size_t>{
        static std::string print(const size_t& val) {
            std::stringstream str;
            str << "size_t(" << val << ")";
            return str.str();
        }
    };

    template <typename First, typename Second>
    struct pretty_printer<std::pair<First, Second>>{
        using T = std::pair<First, Second>;
        static std::string print(const T& val) {
            std::stringstream str;
            str << type_printer<T>::print()
                << "(\n\t" << pretty_printer<First>::print(val.first)
                << ",\n\t" << pretty_printer<Second>::print(val.second)
                << ")";
            return str.str();
        }
    };

    template <typename T>
    struct type_printer{
        static std::string print() {
            return typeid(T).name();
        }
    };

    template <>
    struct type_printer<float>{
        static std::string print() {
            return "float";
        }
    };

    template <>
    struct type_printer<double>{
        static std::string print() {
            return "double";
        }
    };

    template <>
    struct type_printer<size_t>{
        static std::string print() {
            return "size_t";
        }
    };

    template <>
    struct type_printer<int>{
        static std::string print() {
            return "int";
        }
    };

    // std::pair printer
    template <typename First, typename Second>
    struct type_printer<std::pair<First, Second>>{
        static std::string print() {
            std::stringstream str;
            str << "std::pair<" << type_printer<First>::print()
                << ", " << type_printer<Second>::print() << ">";
            return str.str();
        }
    };
} // namespace util

} // namespace memory
} // namespace arb

