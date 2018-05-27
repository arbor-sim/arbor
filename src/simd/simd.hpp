#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <simd/implbase.hpp>
#include <simd/generic.hpp>
#include <simd/native.hpp>

namespace arb {
namespace simd {

namespace simd_detail {
    template <typename Impl>
    struct simd_impl;

    template <typename Impl>
    struct simd_mask_impl;
}

namespace simd_detail {
    template <typename Impl, typename V>
    struct indirect_expression {
        V* p;
        typename simd_traits<Impl>::vector_type index;
        index_constraint constraint;

        indirect_expression() = default;
        indirect_expression(V* p, const simd_impl<Impl>& index_simd, index_constraint constraint):
            p(p), index(index_simd.value_), constraint(constraint)
        {}

        // Simple assignment included for consistency with compound assignment interface.

        template <typename Other>
        indirect_expression& operator=(const simd_impl<Other>& s) {
            s.copy_to(*this);
            return *this;
        }

        // Compound assignment (currently only addition and subtraction!):

        template <typename Other>
        indirect_expression& operator+=(const simd_impl<Other>& s) {
            simd_impl<Other>::compound_indexed_add(tag<Impl>{}, s.value_, p, index, constraint);
            return *this;
        }

        template <typename Other>
        indirect_expression& operator-=(const simd_impl<Other>& s) {
            simd_impl<Other>::compound_indexed_add(tag<Impl>{}, (-s).value_, p, index, constraint);
            return *this;
        }
    };

    template <typename Impl>
    struct simd_impl {
        // Type aliases:
        //
        //     scalar_type           internal value type in one simd lane,
        //     simd_mask             simd_mask_impl specialization represeting comparison results,
        //
        //     vector_type           underlying representation,
        //     mask_type             underlying representation for mask.

        using scalar_type = typename simd_traits<Impl>::scalar_type;
        using simd_mask   = simd_mask_impl<typename simd_traits<Impl>::mask_impl>;

    protected:
        using vector_type = typename simd_traits<Impl>::vector_type;
        using mask_type   = typename simd_traits<typename simd_traits<Impl>::mask_impl>::vector_type;

    public:
        static constexpr unsigned width = simd_traits<Impl>::width;

        template <typename Other>
        friend struct simd_impl;

        template <typename Other, typename V>
        friend struct indirect_expression;

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

        // Construct from scalar values in memory with mask.
        explicit simd_impl(const scalar_type* p, const simd_mask& m) {
            value_ = Impl::copy_from_masked(p, m.value_);
        }

        // Construct from const array ref with mask.
        explicit simd_impl(const scalar_type (&a)[width], const simd_mask& m) {
            value_ = Impl::copy_from_masked(a, m.value_);
        }

        // Construct from a different SIMD value by casting.
        template <typename Other, typename = typename std::enable_if<width==simd_traits<Other>::width>::type>
        explicit simd_impl(const simd_impl<Other>& x) {
            value_ = Impl::cast_from(tag<Other>{}, x.value_);
        }

        // Construct from indirect expression (gather).
        template <typename IndexImpl, typename = typename std::enable_if<width==simd_traits<IndexImpl>::width>::type>
        explicit simd_impl(indirect_expression<IndexImpl, scalar_type> pi) {
            copy_from(pi);
        }

        template <typename IndexImpl, typename = typename std::enable_if<width==simd_traits<IndexImpl>::width>::type>
        explicit simd_impl(indirect_expression<IndexImpl, const scalar_type> pi) {
            copy_from(pi);
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

        template <typename IndexImpl, typename = typename std::enable_if<width==simd_traits<IndexImpl>::width>::type>
        void copy_to(indirect_expression<IndexImpl, scalar_type> pi) const {
            Impl::scatter(tag<IndexImpl>{}, value_, pi.p, pi.index);
        }

        void copy_from(const scalar_type* p) {
            value_ = Impl::copy_from(p);
        }

        template <typename IndexImpl, typename = typename std::enable_if<width==simd_traits<IndexImpl>::width>::type>
        void copy_from(indirect_expression<IndexImpl, scalar_type> pi) {
            switch (pi.constraint) {
                case index_constraint::none:
                {
                    value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index);
                }
                break;
                case index_constraint::independent:
                {
                    value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index);
                }
                break;
                case index_constraint::contiguous:
                {
                    scalar_type* p = IndexImpl::element0(pi.index) + pi.p;
                    value_ = Impl::copy_from(p);
                }
                break;
                case index_constraint::constant:
                {
                    value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index);
                }
                break;
            }
        }

        template <typename IndexImpl, typename = typename std::enable_if<width==simd_traits<IndexImpl>::width>::type>
        void copy_from(indirect_expression<IndexImpl, const scalar_type> pi) {
            switch (pi.constraint) {
                case index_constraint::none:
                {
                    value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index);
                }
                break;
                case index_constraint::independent:
                {
                    value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index);
                }
                break;
                case index_constraint::contiguous:
                {
                    const scalar_type* p = IndexImpl::element0(pi.index) + pi.p;
                    value_ = Impl::copy_from(p);
                }
                break;
                case index_constraint::constant:
                {
                    value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index);
                }
                break;
            }

        }

        template <typename ImplIndex>
        static void compound_indexed_add(tag<ImplIndex> tag, const vector_type& s, scalar_type* p, const typename ImplIndex::vector_type& index, index_constraint constraint) {
            switch (constraint) {
                case index_constraint::none:
                {
                    typename ImplIndex::scalar_type o[width];
                    ImplIndex::copy_to(index, o);

                    scalar_type a[width];
                    Impl::copy_to(s, a);

                    for (unsigned i = 0; i<width; ++i) {
                        p[o[i]] += a[i];
                    }
                }
                break;
                case index_constraint::independent:
                {
                    vector_type v = Impl::add(Impl::gather(tag, p, index), s);
                    Impl::scatter(tag, v, p, index);
                }
                break;
                case index_constraint::contiguous:
                {
                    p += ImplIndex::element0(index);
                    vector_type v = Impl::add(Impl::copy_from(p), s);
                    Impl::copy_to(v, p);
                }
                break;
                case index_constraint::constant:
                {
                    p += ImplIndex::element0(index);
                    *p += Impl::reduce_add(s);
                }
                break;
            }
        }


        // Arithmetic operations: +, -, *, /, fma.

        simd_impl operator-() const {
            return wrap(Impl::negate(value_));
        }

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

        // Reductions (horizontal operations).

        scalar_type sum() const {
            return Impl::reduce_add(value_);
        }

        // Masked assignment (via where expressions).

        struct where_expression {
            where_expression(const where_expression&) = default;
            where_expression& operator=(const where_expression&) = delete;

            where_expression(const simd_mask& m, simd_impl& s):
                mask_(m), data_(s) {}

            where_expression& operator=(scalar_type v) {
                data_.value_ = Impl::ifelse(mask_.value_, simd_impl(v).value_, data_.value_);
                return *this;
            }

            where_expression& operator=(const simd_impl& v) {
                data_.value_ = Impl::ifelse(mask_.value_, v.value_, data_.value_);
                return *this;
            }

            void copy_to(scalar_type* p) const {
                Impl::copy_to_masked(data_.value_, p, mask_.value_);
            }

            void copy_from(const scalar_type* p) {
                data_.value_ = Impl::copy_from_masked(data_.value_, p, mask_.value_);
            }

            // Gather and scatter.

            template <typename IndexImpl, typename = typename std::enable_if<width==simd_traits<IndexImpl>::width>::type>
            void copy_from(indirect_expression<IndexImpl, scalar_type> pi) {
                data_.value_ = Impl::gather(tag<IndexImpl>{}, data_.value_, pi.p, pi.index, mask_.value_);
            }

            template <typename IndexImpl, typename = typename std::enable_if<width==simd_traits<IndexImpl>::width>::type>
            void copy_to(indirect_expression<IndexImpl, scalar_type> pi) const {
                Impl::scatter(tag<IndexImpl>{}, data_.value_, pi.p, pi.index, mask_.value_);
            }

        private:
            const simd_mask& mask_;
            simd_impl& data_;
        };

        // Maths functions are implemented as top-level functions; declare as friends
        // for access to `wrap` and to enjoy ADL, allowing implicit conversion from
        // scalar_type in binary operation arguments.

        friend simd_impl abs(const simd_impl& s) {
            return simd_impl::wrap(Impl::abs(s.value_));
        }

        friend simd_impl sin(const simd_impl& s) {
            return simd_impl::wrap(Impl::sin(s.value_));
        }

        friend simd_impl cos(const simd_impl& s) {
            return simd_impl::wrap(Impl::cos(s.value_));
        }

        friend simd_impl exp(const simd_impl& s) {
            return simd_impl::wrap(Impl::exp(s.value_));
        }

        friend simd_impl log(const simd_impl& s) {
            return simd_impl::wrap(Impl::log(s.value_));
        }

        friend simd_impl expm1(const simd_impl& s) {
            return simd_impl::wrap(Impl::expm1(s.value_));
        }

        friend simd_impl exprelr(const simd_impl& s) {
            return simd_impl::wrap(Impl::exprelr(s.value_));
        }

        friend simd_impl pow(const simd_impl& s, const simd_impl& t) {
            return simd_impl::wrap(Impl::pow(s.value_, t.value_));
        }

        friend simd_impl min(const simd_impl& s, const simd_impl& t) {
            return simd_impl::wrap(Impl::min(s.value_, t.value_));
        }

        friend simd_impl max(const simd_impl& s, const simd_impl& t) {
            return simd_impl::wrap(Impl::max(s.value_, t.value_));
        }

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

        // Make mask from corresponding bits of integer.

        static simd_mask_impl unpack(unsigned long long bits) {
            return simd_mask_impl::wrap(Impl::mask_unpack(bits));
        }

    private:
        simd_mask_impl(const vector_type& v): base(v) {}

        template <class> friend struct simd_impl;

        static simd_mask_impl wrap(const vector_type& v) {
            simd_mask_impl m;
            m.value_ = v;
            return m;
        }
    };

    template <typename To>
    struct simd_cast_impl {};

    template <typename ImplTo>
    struct simd_cast_impl<simd_impl<ImplTo>> {
        static constexpr unsigned N = simd_traits<ImplTo>::width;
        using scalar_type = typename simd_traits<ImplTo>::scalar_type;

        template <typename ImplFrom, typename = typename std::enable_if<N==simd_traits<ImplFrom>::width>::type>
        static simd_impl<ImplTo> cast(const simd_impl<ImplFrom>& v) {
            return simd_impl<ImplTo>(v);
        }

        static simd_impl<ImplTo> cast(const std::array<scalar_type, N>& a) {
            return simd_impl<ImplTo>(a.data());
        }
    };

    template <typename V, std::size_t N>
    struct simd_cast_impl<std::array<V, N>> {
        template <
            typename ImplFrom,
            typename = typename std::enable_if<
                N==simd_traits<ImplFrom>::width &&
                std::is_same<V, typename simd_traits<ImplFrom>::scalar_type>::value
            >::type
        >
        static std::array<V, N> cast(const simd_impl<ImplFrom>& s) {
            std::array<V, N> a;
            s.copy_to(a.data());
            return a;
        }
    };
} // namespace simd_detail

namespace simd_abi {
    // Note: `simd_abi::native` template class defined in `simd/native.hpp`,
    // `simd_abi::generic` in `simd/generic.hpp`.

    template <typename Value, unsigned N>
    struct default_abi {
        using type = typename std::conditional<
            std::is_same<void, typename native<Value, N>::type>::value,
            typename generic<Value, N>::type,
            typename native<Value, N>::type>::type;
    };
}

template <typename Value, unsigned N, template <class, unsigned> class Abi = simd_abi::default_abi>
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

// Casting is dispatched to simd_cast_impl in order to handle conversions to
// and from std::array.

template <typename To, typename From>
To simd_cast(const From& s) {
    return simd_detail::simd_cast_impl<To>::cast(s);
}

// Gather/scatter indexed memory specification.

template <
    typename IndexImpl,
    typename PtrLike,
    typename V = typename std::remove_reference<decltype(*std::declval<PtrLike>())>::type
>
simd_detail::indirect_expression<IndexImpl, V> indirect(
    PtrLike p,
    const simd_detail::simd_impl<IndexImpl>& index,
    index_constraint constraint = index_constraint::none)
{
    return simd_detail::indirect_expression<IndexImpl, V>(p, index, constraint);
}


} // namespace simd
} // namespace arb
