#pragma once

#include <type_traits>

#include <util/simd/generic.hpp>
#include <util/simd/native.hpp>

namespace arb {

namespace simd_detail {
    template <typename Impl>
    struct simd_mask_impl;

    template <typename Impl>
    struct simd_impl {
        // Type aliases:
        //
        //     vector_type           underlying representation,
        //     scalar_type           internal value type in one simd lane,
        //     simd_mask             simd_mask_impl specialization represeting comparison results.

        using vector_type = typename Impl::vector_type;
        using scalar_type = typename Impl::scalar_type;
        using simd_mask   = simd_mask_impl<typename Impl::mask_impl>;

        static constexpr unsigned width = Impl::width;

        simd_impl() = default;

        // Construct by filling with scalar value.
        simd_impl(const scalar_type& x) {
            value_ = Impl::broadcast(x);
        }

        // Construct from scalar values in memory.
        explicit simd_impl(const scalar_type* p) {
            value_ = Impl::copy_from(p);
        }

        // Construct from const array ref or std::array.
        explicit simd_impl(const scalar_type (&a)[width]) {
            value_ = Impl::copy_from(&a[0]);
        }

        // Copy constructor.
        simd_impl(const simd_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
        }

        // Scalar asssignment fills vector.
        simd_impl& operator=(scalar_type x) {
            value_ = Impl::broadcast(x);
            return *this;
        }

        // Copy assignment.
        simd_impl& operator=(const simd_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
            return *this;
        }

        // Read/write operations: copy_to, copy_from.

        void copy_to(scalar_type* p) const {
            Impl::copy_to(value_, p);
        }

        void copy_from(const scalar_type* p) {
            value_ = Impl::copy_from(p);
        }

        // Arithmetic operations: +, -, *, /, fma.

        friend simd_impl operator+(const simd_impl& a, simd_impl b) {
            return Impl::add(a.value_, b.value_);
        }

        friend simd_impl operator-(const simd_impl& a, simd_impl b) {
            return Impl::sub(a.value_, b.value_);
        }

        friend simd_impl operator*(const simd_impl& a, simd_impl b) {
            return Impl::mul(a.value_, b.value_);
        }

        friend simd_impl operator/(const simd_impl& a, simd_impl b) {
            return Impl::div(a.value_, b.value_);
        }

        friend simd_impl fma(const simd_impl& a, simd_impl b, simd_impl c) {
            return Impl::fma(a.value_, b.value_, c.value_);
        }

        // Lane-wise relational operations.

        friend simd_mask operator==(const simd_impl& a, const simd_impl& b) {
            return Impl::cmp_eq(a.value_, b.value_);
        }

        friend simd_mask operator!=(const simd_impl& a, const simd_impl& b) {
            return Impl::cmp_neq(a.value_, b.value_);
        }

        friend simd_mask operator<=(const simd_impl& a, const simd_impl& b) {
            return Impl::cmp_leq(a.value_, b.value_);
        }

        friend simd_mask operator<(const simd_impl& a, const simd_impl& b) {
            return Impl::cmp_lt(a.value_, b.value_);
        }

        friend simd_mask operator>=(const simd_impl& a, const simd_impl& b) {
            return Impl::cmp_geq(a.value_, b.value_);
        }

        friend simd_mask operator>(const simd_impl& a, const simd_impl& b) {
            return Impl::cmp_gt(a.value_, b.value_);
        }

        // Compound assignment operations: +=, -=, *=, /=.

        simd_impl& operator+=(const simd_impl& x) {
            value_ = Impl::add(value_, x.value_);
            return *this;
        }

        simd_impl& operator-=(const simd_impl& x) {
            value_ = Impl::sub(value_, x.value_);
            return *this;
        }

        simd_impl& operator*=(const simd_impl& x) {
            value_ = Impl::mul(value_, x.value_);
            return *this;
        }

        simd_impl& operator/=(const simd_impl& x) {
            value_ = Impl::div(value_, x.value_);
            return *this;
        }

        // Array subscript operations.

        struct reference {
            reference() = delete;
            reference(const reference&) = default;
            reference& operator=(const reference&) = delete;

            reference(vector_type& value, int i):
                ptr_(&value), i(i) {}

            reference& operator=(scalar_type v) && {
                Impl::set_element(*ptr_, i, v);
                return *this;
            }

            operator scalar_type() const {
                return Impl::element(*ptr_, i);
            }

            vector_type* ptr_;
            int i;
        };

        reference operator[](int i) {
            return reference(value_, i);
        }

        scalar_type operator[](int i) const {
            return Impl::element(value_, i);
        }

        // Masked assignment (via where expressions).

        struct where_expression {
            where_expression(const where_expression&) = default;
            where_expression& operator=(const where_expression&) = delete;

            where_expression(const simd_mask& m, simd_impl& v):
                mask_(m), data_(v) {}

            where_expression& operator=(scalar_type v) {
                data_ = Impl::select(mask_.value_, data_.value_, simd_impl(v).value_);
                return *this;
            }

            where_expression& operator=(const simd_impl& v) {
                data_ = Impl::select(mask_.value_, data_.value_, v.value_);
                return *this;
            }

        private:
            const simd_mask& mask_;
            simd_impl& data_;
        };

    protected:
        vector_type value_;

        simd_impl(const vector_type& x) {
            value_ = x;
        }
    };

    template <typename Impl>
    struct simd_mask_impl: simd_impl<Impl> {
        using base = simd_impl<Impl>;
        using typename base::vector_type;
        using typename base::scalar_type;
        using base::value_;

        simd_mask_impl() = default;

        // Construct by filling with scalar value.
        simd_mask_impl(bool x) {
            value_ = Impl::broadcast(x);
        }

        // Scalar asssignment fills vector.
        simd_mask_impl& operator=(bool x) {
            value_ = Impl::broadcast(x);
            return *this;
        }

        // Copy assignment.
        simd_mask_impl& operator=(const simd_mask_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
            return *this;
        }

        // Array subscript operations.

        struct reference {
            reference() = delete;
            reference(const reference&) = default;
            reference& operator=(const reference&) = delete;

            reference(vector_type& value, int i):
                ptr_(&value), i(i) {}

            reference& operator=(bool v) && {
                Impl::set_element(*ptr_, i, v);
                return *this;
            }

            operator scalar_type() const {
                return Impl::bool_element(*ptr_, i);
            }

            vector_type* ptr_;
            int i;
        };

        reference operator[](int i) {
            return reference(value_, i);
        }

        bool operator[](int i) const {
            return Impl::element(value_, i);
        }

        // Logical operations.

        simd_mask_impl operator!() const {
            return Impl::logical_not(value_);
        }

        friend simd_mask_impl operator&&(const simd_mask_impl& a, const simd_mask_impl& b) {
            return Impl::logical_and(a.value_, b.value_);
        }

        friend simd_mask_impl operator||(const simd_mask_impl& a, const simd_mask_impl& b) {
            return Impl::logical_or(a.value_, b.value_);
        }

    protected:
        simd_mask_impl(const vector_type& v): base(v) {}
    };
} // namespace simd_detail

namespace simd_abi {
    template <typename Value, unsigned N>
    struct generic {
        using type = simd_detail::generic<Value, N>;
    };

    // Note: `simd_abi::native` template class defined in `simd/native.hpp`.

    template <typename Value, unsigned N>
    struct default_abi {
        using type = typename std::conditional<
            std::is_same<void, typename native<Value, N>::type>::value,
            typename generic<Value, N>::type,
            typename native<Value, N>::type>::type;
    };
}

template <typename Value, unsigned N, template <class, unsigned> typename Abi = simd_abi::default_abi>
using simd = simd_detail::simd_impl<typename Abi<Value, N>::type>;

template <typename Value, unsigned N>
using simd_mask = typename simd<Value, N>::simd_mask;

template <typename Simd>
using where_expression = typename Simd::where_expression;

template <typename Simd>
where_expression<Simd> where(const typename Simd::simd_mask& m, Simd& v) {
    return where_expression<Simd>(m, v);
}

} // namespace arb
