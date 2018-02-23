#pragma once

#include <type_traits>

#include <util/simd/generic.hpp>
#include <util/simd/native.hpp>

namespace arb {

namespace simd_detail {
    // Fallback implementations for gather, scatter, simd_cast: specialize per ABI impl.

    template <typename Impl, typename ImplIndex>
    struct gather_impl {
        using vector_type = typename Impl::vector_type;
        using scalar_type = typename Impl::scalar_type;
        using index_type = typename ImplIndex::vector_type;

        static vector_type gather(const scalar_type* p, const index_type& index) {
            constexpr unsigned N = Impl::width;

            typename ImplIndex::scalar_type o[N];
            scalar_type data[N];

            ImplIndex::copy_to(index, o);

            for (unsigned i = 0; i<N; ++i) {
                data[i] = p[o[i]];
            }
            return Impl::copy_from(data);
        }
    };

    template <typename Impl, typename ImplIndex>
    struct masked_gather_impl {
        using vector_type = typename Impl::vector_type;
        using scalar_type = typename Impl::scalar_type;
        using index_type = typename ImplIndex::vector_type;

        using mask_impl   = typename Impl::mask_impl;
        using mask_type   = typename mask_impl::vector_type;

        static vector_type gather(vector_type a, const scalar_type* p, const index_type& index, const mask_type& mask) {
            constexpr unsigned N = Impl::width;

            typename ImplIndex::scalar_type o[N];
            bool m[N];
            scalar_type data[N];

            Impl::copy_to(a, data);
            ImplIndex::copy_to(index, o);
            mask_impl::mask_copy_to(mask, m);

            for (unsigned i = 0; i<N; ++i) {
                if (m[i]) { data[i] = p[o[i]]; }
            }
            a = Impl::copy_from(data);
            return a;
        }
    };

    template <typename Impl, typename ImplIndex>
    struct scatter_impl {
        using vector_type = typename Impl::vector_type;
        using scalar_type = typename Impl::scalar_type;
        using index_type = typename ImplIndex::vector_type;

        static void scatter(vector_type s, const scalar_type* p, const index_type& index) {
            constexpr unsigned N = Impl::width;

            typename ImplIndex::scalar_type o[N];
            scalar_type data[N];

            Impl::copy_to(s, data);
            ImplIndex::copy_to(index, o);

            for (unsigned i = 0; i<N; ++i) {
                p[o[i]] = data[i];
            }
        }
    };

    template <typename Impl, typename ImplIndex>
    struct masked_scatter_impl {
        using vector_type = typename Impl::vector_type;
        using scalar_type = typename Impl::scalar_type;
        using index_type = typename ImplIndex::vector_type;

        using mask_impl   = typename Impl::mask_impl;
        using mask_type   = typename mask_impl::vector_type;

        static void scatter(vector_type s, const scalar_type* p, const index_type& index, const mask_type& mask) {
            constexpr unsigned N = Impl::width;

            typename ImplIndex::scalar_type o[N];
            bool m[N];
            scalar_type data[N];

            Impl::copy_to(s, data);
            ImplIndex::copy_to(index, o);
            mask_impl::mask_copy_to(mask, m);

            for (unsigned i = 0; i<N; ++i) {
                if (m[i]) { p[o[i]] = data[i]; }
            }
        }
    };

    template <typename Impl>
    struct simd_mask_impl;

    template <typename Impl>
    struct simd_impl {
        // Type aliases:
        //
        //     vector_type           underlying representation,
        //     mask_type             underlying representation for mask,
        //     scalar_type           internal value type in one simd lane,
        //     simd_mask             simd_mask_impl specialization represeting comparison results.

        using vector_type = typename Impl::vector_type;
        using scalar_type = typename Impl::scalar_type;
        using simd_mask   = simd_mask_impl<typename Impl::mask_impl>;
        using mask_type   = typename Impl::mask_impl::vector_type;

        static constexpr unsigned width = Impl::width;


        template <typename Other>
        friend class simd_impl;

        simd_impl() = default;

        // Construct by filling with scalar value.
        simd_impl(const scalar_type& x) {
            value_ = Impl::broadcast(x);
        }

        // Construct from scalar values in memory.
        explicit simd_impl(const scalar_type* p) {
            value_ = Impl::copy_from(p);
        }

        // Construct from const array ref.
        explicit simd_impl(const scalar_type (&a)[width]) {
            value_ = Impl::copy_from(a);
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
            return simd_impl::wrap(Impl::add(a.value_, b.value_));
        }

        friend simd_impl operator-(const simd_impl& a, simd_impl b) {
            return simd_impl::wrap(Impl::sub(a.value_, b.value_));
        }

        friend simd_impl operator*(const simd_impl& a, simd_impl b) {
            return simd_impl::wrap(Impl::mul(a.value_, b.value_));
        }

        friend simd_impl operator/(const simd_impl& a, simd_impl b) {
            return simd_impl::wrap(Impl::div(a.value_, b.value_));
        }

        friend simd_impl fma(const simd_impl& a, simd_impl b, simd_impl c) {
            return simd_impl::wrap(Impl::fma(a.value_, b.value_, c.value_));
        }

        // Lane-wise relational operations.

        friend simd_mask operator==(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_eq(a.value_, b.value_));
        }

        friend simd_mask operator!=(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_neq(a.value_, b.value_));
        }

        friend simd_mask operator<=(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_leq(a.value_, b.value_));
        }

        friend simd_mask operator<(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_lt(a.value_, b.value_));
        }

        friend simd_mask operator>=(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_geq(a.value_, b.value_));
        }

        friend simd_mask operator>(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_gt(a.value_, b.value_));
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

        // Gather (dispatch to simd_detail::gather_impl or simd_detail::masked_gather_impl).

        template <typename IndexImpl, typename = typename std::enable_if<width==IndexImpl::width>::type>
        void gather(const scalar_type* p, const simd_impl<IndexImpl>& index) {
            value_ = gather_impl<Impl, IndexImpl>::gather(p, index.value_);
        }

        template <typename IndexImpl, typename = typename std::enable_if<width==IndexImpl::width>::type>
        void gather(const scalar_type* p, const simd_impl<IndexImpl>& index, const simd_mask& mask) {
            value_ = masked_gather_impl<Impl, IndexImpl>::gather(value_, p, index.value_, mask.value_);
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

    private:
        static simd_impl wrap(const vector_type& v) {
            simd_impl s;
            s.value_ = v;
            return s;
        }

        static simd_mask mask(const mask_type& v) {
            simd_mask m;
            m.value_ = v;
            return m;
        }
    };

    template <typename Impl>
    struct simd_mask_impl: simd_impl<Impl> {
        using base = simd_impl<Impl>;
        using typename base::vector_type;
        using typename base::scalar_type;
        using base::width;
        using base::value_;

        simd_mask_impl() = default;

        // Construct by filling with scalar value.
        simd_mask_impl(bool b) {
            value_ = Impl::mask_broadcast(b);
        }

        // Scalar asssignment fills vector.
        simd_mask_impl& operator=(bool b) {
            value_ = Impl::mask_broadcast(b);
            return *this;
        }

        // Construct from bool values in memory.
        explicit simd_mask_impl(const bool* y) {
            value_ = Impl::mask_copy_from(y);
        }

        // Construct from const array ref.
        explicit simd_mask_impl(const bool (&a)[width]) {
            value_ = Impl::mask_copy_from(&a[0]);
        }

        // Copy assignment.
        simd_mask_impl& operator=(const simd_mask_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
            return *this;
        }

        // Read/write bool operations: copy_to, copy_from.

        void copy_to(bool* w) const {
            Impl::mask_copy_to(value_, w);
        }

        void copy_from(const bool* y) {
            value_ = Impl::mask_copy_from(y);
        }

        // Array subscript operations.

        struct reference {
            reference() = delete;
            reference(const reference&) = default;
            reference& operator=(const reference&) = delete;

            reference(vector_type& value, int i):
                ptr_(&value), i(i) {}

            reference& operator=(bool v) && {
                Impl::mask_set_element(*ptr_, i, v);
                return *this;
            }

            operator bool() const {
                return Impl::mask_element(*ptr_, i);
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
            return simd_mask_impl::wrap(Impl::logical_not(value_));
        }

        friend simd_mask_impl operator&&(const simd_mask_impl& a, const simd_mask_impl& b) {
            return simd_mask_impl::wrap(Impl::logical_and(a.value_, b.value_));
        }

        friend simd_mask_impl operator||(const simd_mask_impl& a, const simd_mask_impl& b) {
            return simd_mask_impl::wrap(Impl::logical_or(a.value_, b.value_));
        }

    private:
        simd_mask_impl(const vector_type& v): base(v) {}

        template <class> friend class simd_impl;

        static simd_mask_impl wrap(const vector_type& v) {
            simd_mask_impl m;
            m.value_ = v;
            return m;
        }
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

template <typename>
struct is_simd: std::false_type {};

template <typename Impl>
struct is_simd<simd_detail::simd_impl<Impl>>: std::true_type {};

} // namespace arb
