#pragma once

// The ordered forest data structure represents zero or more trees with an
// arbitrary number of children per node, where the trees and children have a
// well-defined ordering.
//
// This implementation presents a 'forward' interface, along the lines of
// `std::forward_list`. Given an iterator pointing to a particular node in the
// forest, one can directly obtain an iterator to its parent, its first child,
// or its next sibling, but not its previous sibling.
//
// Common iterator methods:
//
//     `parent()`: return an iterator to the node's parent.
//     `next()`: return an iterator to the node's next sibling.
//     `child()`: return an iterator to the node's first child.
//     `preorder_next()`: return an iterator to the next node in pre-order.
//     `postorder_next()`: return an iterator to the next node in post-order.
//
// Note that these methods are not covariant: they return 'base' iterators
// that can be implicitly converted to any of the derived iterators described
// below.
//
// A default-constructed iterator points to no node, and plays the role of
// an `end` iterator.
//
// The derived iterators differ only in the semantics of `operator++`:
//
//     `sibling_iterator`: proceed to next sibling.
//     `preorder_iterator`: proceed to next node in pre-order.
//     `postorder_iterator`: proceed to next node in post-order.
//
// Any of the iterators can be converted implicitly to any const iterator, and
// any of the non-const iterators can be converted implicitly to any non-const
// iterator.
//
// The iterators provided by `begin()` and `end()` are pre-order iterators.
// The other ordered forest iterator methods are:
//
//     `child_begin`: return a sibling iterator starting at the child of the given node.
//     `root_begin`: return a sibling iterator to the first tree.
//     `postorder_begin`: return a post-order iterator to the first node in post-order.
//     `preorder_begin`: return a post-order iterator to the first node in pre-order.
//
// Note that all ordered forest iterators are forward iterators.
//
// Ordered forests can be constructed from an intializer list with the aid of
// the `ordered_forest_builder` helper class. Each item in the initalizer list
// is either a value (corresponding to a node without children), or a pair
// { value, IL } where value denotes the value at the node, and IL is an
// initializer list with the same semantics, defining the node's children.
//
// Mutation operations are of the following types:
//
// * Insertions: add a new node as next sibling, first child, or first tree.
// * Grafts: splice a forest between a node and its next sibling, or before its first child.
// * Erasures: remove a node, replacing it with its children.
// * Cuts: remove a sub-tree, presenting it as a new forest.
//
// These methods are described in more detail within the class definition below.
//
// The ordered forest implementation is allocator aware.

#include <type_traits>
#include <iterator>
#include <memory>
#include <utility>

namespace arb {
namespace util {

template <typename V, typename Allocator>
struct ordered_forest_builder;

template <typename V, typename Allocator = std::allocator<V>>
class ordered_forest {
    struct node {
        V* item_ = nullptr;
        node* parent_ = nullptr;
        node* child_ = nullptr;
        node* next_ = nullptr;
    };

    using node_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<node>;
    using item_alloc_traits = std::allocator_traits<Allocator>;
    using node_alloc_traits = std::allocator_traits<node_alloc_t>;

public:
    using value_type = V;
    using allocator_type = Allocator;
    using size_type = std::size_t;

    struct iterator_base {
        node* n_ = nullptr;

        iterator_base() = default;
        explicit iterator_base(node* n): n_(n) {}
        explicit operator bool() const { return n_; }
    };

    // Constructor overloads for iterators permit construction of any const
    // iterator from any other iterator, or of any mutable iterator from any
    // other mutable iterator.

    template <bool const_flag>
    struct iterator_mc: iterator_base {
        using pointer = std::conditional_t<const_flag, const V*, V*>;
        using reference = std::conditional_t<const_flag, const V&, V&>;
        using value_type = V;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator_mc() = default;

        template <bool flag = const_flag, typename std::enable_if_t<flag, int> = 0>
        iterator_mc(const iterator_mc<false>& i): iterator_base(i.n_) {}

        iterator_mc parent() const { return iterator_mc{n_? n_->parent_: nullptr}; }
        iterator_mc next() const { return iterator_mc{n_? n_->next_: nullptr}; }
        iterator_mc child() const { return iterator_mc{n_? n_->child_: nullptr}; }

        bool operator==(const iterator_base& a) const { return n_ == a.n_; }
        bool operator!=(const iterator_base& a) const { return n_ != a.n_; }

        iterator_mc preorder_next() const {
            if (!n_) return {};
            if (n_->child_) return iterator_mc{n_->child_};

            node* x = n_;
            while (x && !x->next_) x = x->parent_;
            return iterator_mc{x? x->next_: nullptr};
        }

        iterator_mc postorder_next() const {
            if (!n_) return {};
            if (n_->next_) {
                node* x = n_->next_;
                while (node* c = x->child_) x = c;
                return iterator_mc{x};
            }
            else return parent();
        }

        reference operator*() const { return *n_->item_; }
        pointer operator->() const { return n_->item_; }

    protected:
        friend ordered_forest;

        explicit iterator_mc(node* n): iterator_base(n) {}
        using iterator_base::n_;
    };

    template <bool const_flag>
    struct sibling_iterator_mc: iterator_mc<const_flag> {
        sibling_iterator_mc() = default;
        sibling_iterator_mc(const iterator_mc<const_flag>& i): iterator_mc<const_flag>(i) {}

        sibling_iterator_mc& operator++() { return *this = this->next(); }
        sibling_iterator_mc operator++(int) { auto p = *this; return ++*this, p; }
    };

    using sibling_iterator = sibling_iterator_mc<false>;
    using const_sibling_iterator = sibling_iterator_mc<true>;

    template <bool const_flag>
    struct preorder_iterator_mc: iterator_mc<const_flag> {
        preorder_iterator_mc() = default;
        preorder_iterator_mc(const iterator_mc<const_flag>& i): iterator_mc<const_flag>(i) {}

        preorder_iterator_mc& operator++() { return *this = this->preorder_next(); }
        preorder_iterator_mc operator++(int) { auto p = *this; return ++*this, p; }
    };

    using preorder_iterator = preorder_iterator_mc<false>;
    using const_preorder_iterator = preorder_iterator_mc<true>;

    template <bool const_flag>
    struct postorder_iterator_mc: iterator_mc<const_flag> {
        postorder_iterator_mc() = default;
        postorder_iterator_mc(const iterator_mc<const_flag>& i): iterator_mc<const_flag>(i) {}

        postorder_iterator_mc& operator++() { return *this = this->postorder_next(); }
        postorder_iterator_mc operator++(int) { auto p = *this; return ++*this, p; }
    };

    using postorder_iterator = postorder_iterator_mc<false>;
    using const_postorder_iterator = postorder_iterator_mc<true>;

    bool empty() const { return !first_; }

    // Note: size is O(n) in the number of elements n.
    std::size_t size() const {
        std::size_t n = 0;
        for (const auto& _: *this) (void)_, ++n;
        return n;
    }

    sibling_iterator child_begin(const iterator_mc<false>& i) { return sibling_iterator{i.child()}; }
    const_sibling_iterator child_begin(const iterator_mc<true>& i) const { return const_sibling_iterator{i.child()}; }

    sibling_iterator child_end(const iterator_base&) { return {}; }
    const_sibling_iterator child_end(const iterator_base&) const { return {}; }

    sibling_iterator root_begin() { return sibling_iterator{first_else_end()}; }
    const_sibling_iterator root_begin() const { return const_sibling_iterator{first_else_end()}; }

    sibling_iterator root_end() { return sibling_iterator{}; }
    const_sibling_iterator root_end() const { return const_sibling_iterator{}; }

    postorder_iterator postorder_begin() { return postorder_iterator{first_leaf()}; }
    const_postorder_iterator postorder_begin() const { return const_postorder_iterator{first_leaf()}; }

    postorder_iterator postorder_end() { return {}; }
    const_postorder_iterator postorder_end() const { return {}; }

    preorder_iterator preorder_begin() { return preorder_iterator{first_else_end()}; }
    const_preorder_iterator preorder_begin() const { return const_preorder_iterator{first_else_end()}; }

    preorder_iterator preorder_end() { return {}; }
    const_preorder_iterator preorder_end() const { return {}; }

    // Default iteration is preorder.

    using iterator = preorder_iterator;
    using const_iterator = const_preorder_iterator;

    iterator begin() { return preorder_begin(); }
    iterator end() { return {}; }

    const_iterator begin() const { return preorder_begin(); }
    const_iterator end() const { return {}; }

    const_iterator cbegin() const { return preorder_begin(); }
    const_iterator cend() const { return {}; }

    // Insertion and emplace operations:
    //
    // * All return an iterator to the last inserted node, or an iterator to the referenced
    //   node (or first tree, for push_front) if the collection of inserted items is empty.
    //
    // * The iterator argument may not be an end iterator.
    //
    // Insertion member functions are templated on iterator class in order to return covariantly.

    // Insert/emplace item as next sibling.

    template <typename Iter, typename = std::enable_if_t<std::is_base_of<iterator_mc<false>, Iter>::value>>
    Iter insert_after(const Iter& i, const V& item) {
        return assert_valid(i), splice_impl(i.n_->parent_, i.n_->next_, make_node(item));
    }

    template <typename Iter, typename = std::enable_if_t<std::is_base_of<iterator_mc<false>, Iter>::value>>
    Iter insert_after(const Iter& i, V&& item) {
        return assert_valid(i), splice_impl(i.n_->parent_, i.n_->next_, make_node(std::move(item)));
    }

    template <typename Iter, typename... Args, typename = std::enable_if_t<std::is_base_of<iterator_mc<false>, Iter>::value>>
    Iter emplace_after(const Iter& i, Args&&... args) {
        return assert_valid(i), splice_impl(i.n_->parent_, i.n_->next_, make_node(std::forward<Args>(args)...));
    }

    // Insert trees in forest as next siblings.

    template <typename Iter, typename = std::enable_if_t<std::is_base_of<iterator_mc<false>, Iter>::value>>
    Iter graft_after(const Iter& i, ordered_forest of) {
        assert_valid(i);
        if (of.empty()) return i;

        // Underlying move or copy depends upon allocator equality.
        ordered_forest f(std::move(of), get_allocator());
        node* sp_first = f.first_;
        f.first_ = nullptr;
        return splice_impl(i.n_->parent_, i.n_->next_, sp_first);
    }

    // Insert item as first child.

    template <typename Iter, typename = std::enable_if_t<std::is_base_of<iterator_mc<false>, Iter>::value>>
    Iter push_child(const Iter& i, const V& item) {
        return assert_valid(i), splice_impl(i.n_, i.n_->child_, make_node(item));
    }

    template <typename Iter, typename = std::enable_if_t<std::is_base_of<iterator_mc<false>, Iter>::value>>
    Iter push_child(const Iter& i, V&& item) {
        return assert_valid(i), splice_impl(i.n_, i.n_->child_, make_node(std::move(item)));
    }

    template <typename Iter, typename... Args, typename = std::enable_if_t<std::is_base_of<iterator_mc<false>, Iter>::value>>
    Iter emplace_child(const Iter& i, Args&&... args) {
        return assert_valid(i), splice_impl(i.n_, i.n_->child_, make_node(std::forward<Args>(args)...));
    }

    // Insert trees in forest as first children.

    template <typename Iter, typename = std::enable_if_t<std::is_base_of<iterator_mc<false>, Iter>::value>>
    Iter graft_child(const Iter& i, ordered_forest of) {
        assert_valid(i);
        if (of.empty()) return i;

        // Underlying move or copy depends upon allocator equality.
        ordered_forest f(std::move(of), get_allocator());
        node* sp_first = f.first_;
        f.first_ = nullptr;
        return splice_impl(i.n_, i.n_->child_, sp_first);
    }

    // Insert item as first top-level tree.

    iterator push_front(const V& item) {
        return splice_impl(nullptr, first_, make_node(item));
    }

    iterator push_front(V&& item) {
        return splice_impl(nullptr, first_, make_node(std::move(item)));
    }

    template <typename... Args>
    iterator emplace_front(Args&&... args) {
        return splice_impl(nullptr, first_, make_node(std::forward<Args>(args)...));
    }

    // Insert trees in forest as first top-level children.

    iterator graft_front(ordered_forest of) {
        if (of.empty()) return {};

        // Underlying move or copy depends upon allocator equality.
        ordered_forest f(std::move(of), get_allocator());
        node* sp_first = f.first_;
        f.first_ = nullptr;
        return splice_impl(nullptr, first_, sp_first);
    }

    // Erase and cut operations:
    //
    // * Erase/pop operations replace a node with all of that node's children.
    // * Prune operations remove a whole subtree, and return it as a new ordered forest.

    // Erase/cut next sibling.

    void erase_after(const iterator_mc<false>& i) {
        assert_valid(i.next()), erase_impl(i.n_->next_);
    }

    ordered_forest prune_after(const iterator_mc<false>& i) {
        return assert_valid(i.next()), prune_impl(i.n_->next_);
    }

    // Erase/cut first child.

    void erase_child(const iterator_mc<false>& i) {
        assert_valid(i.child()), erase_impl(i.n_->child_);
    }

    ordered_forest prune_child(const iterator_mc<false>& i) {
        return assert_valid(i.child()), prune_impl(i.n_->child_);
    }

    // Erase/cut root of first tree. Precondition: forest is non-empty.

    void erase_front() {
        assert_nonempty(), erase_impl(first_);
    }

    ordered_forest prune_front() {
        return assert_nonempty(), prune_impl(first_);
    }

    // Access by reference to root of first tree.

    V& front() { return *begin(); }
    const V& front() const { return *begin(); }

    // Comparison:

    bool operator==(const ordered_forest& other) const {
        const_iterator a = begin();
        const_iterator b = other.begin();

        while (a && b) {
            if (!a.child() != !b.child() || !a.next() != !b.next() || !(*a==*b)) return false;
            ++a, ++b;
        }
        return !a && !b;
    }

    bool operator!=(const ordered_forest& other) const {
        return !(*this==other);
    }

    // Constructors, assignment, destructors:

    ordered_forest(const Allocator& alloc = Allocator()) noexcept:
        item_alloc_(alloc),
        node_alloc_(alloc)
    {}

    ordered_forest(ordered_forest&& other) noexcept:
        ordered_forest(std::move(other.item_alloc_))
    {
        first_ = other.first_;
        other.first_ = nullptr;
    }

    ordered_forest(ordered_forest&& other, const Allocator& alloc):
        ordered_forest(alloc)
    {
        if (allocators_equal(other)) {
            first_ = other.first_;
            other.first_ = nullptr;
        }
        else {
            copy_impl(other);
        }
    }

    ordered_forest(std::initializer_list<ordered_forest_builder<V, Allocator>> blist, const Allocator& alloc = Allocator{}):
        ordered_forest(alloc)
    {
        sibling_iterator j;
        for (auto& b: blist) {
            ordered_forest f(b.f_, item_alloc_);
            j = j? graft_after(j, std::move(f)): sibling_iterator(graft_front(std::move(f)));
        }
    }

    ordered_forest(const ordered_forest& other):
        ordered_forest(item_alloc_traits::select_on_container_copy_construction(other.item_alloc_))
    {
        copy_impl(other);
    }

    ordered_forest(const ordered_forest& other, const Allocator& alloc):
        ordered_forest(alloc)
    {
        copy_impl(other);
    }

    ordered_forest& operator=(const ordered_forest& other) {
        if (this==&other) return *this;
        delete_node(first_);

        if (item_alloc_traits::propagate_on_container_copy_assignment::value) {
            item_alloc_ = other.item_alloc_;
        }
        if (node_alloc_traits::propagate_on_container_copy_assignment::value) {
            node_alloc_ = other.node_alloc_;
        }

        copy_impl(other);
        return *this;
    }

    ordered_forest& operator=(ordered_forest&& other) {
        if (this==&other) return *this;
        delete_node(first_);

        if (item_alloc_traits::propagate_on_container_move_assignment::value) {
            item_alloc_ = other.item_alloc_;
        }
        if (node_alloc_traits::propagate_on_container_move_assignment::value) {
            node_alloc_ = other.node_alloc_;
        }

        if (allocators_equal(other)) {
            first_ = other.first_;
            other.first_ = nullptr;
        }
        else {
            copy_impl(other);
        }
        return *this;
    }

    ~ordered_forest() { delete_node(first_); }

    // Swap

    void swap(ordered_forest& other)
        noexcept(
            (item_alloc_traits::propagate_on_container_swap::value && node_alloc_traits::propagate_on_container_swap::value) ||
            (item_alloc_traits::is_always_equal::value && node_alloc_traits::is_always_equal::value)
        )
    {
        using std::swap;
        if (item_alloc_traits::propagate_on_container_swap::value) {
            swap(item_alloc_, other.item_alloc_);
        }
        if (node_alloc_traits::propagate_on_container_swap::value) {
            swap(node_alloc_, other.node_alloc_);
        }

        swap(first_, other.first_);
    }

    friend void swap(ordered_forest& a, ordered_forest& b) {
        a.swap(b);
    }

    // Allocator access

    allocator_type get_allocator() const noexcept { return item_alloc_; }

private:
    Allocator item_alloc_;
    node_alloc_t node_alloc_;
    node* first_ = nullptr;

    bool allocators_equal(const ordered_forest& other) const {
        return item_alloc_==other.item_alloc_ && node_alloc_==other.node_alloc_;
    }

    iterator_mc<false> first_else_end() { return iterator_mc<false>{first_}; }
    iterator_mc<true> first_else_end() const { return iterator_mc<true>{first_}; }

    iterator_mc<false> first_leaf() { return iterator_mc<false>{first_leaf_node()}; }
    iterator_mc<true> first_leaf() const { return iterator_mc<true>{first_leaf_node()}; }

    node* first_leaf_node() const {
        node* n = first_;
        while (n && n->child_) n = n->child_;
        return n;
    }

    // Throw on invalid iterator.
    void assert_valid(iterator_base i) {
        if (!i.n_) throw std::invalid_argument("bad iterator");
    }

    void assert_nonempty() {
        if (!first_) throw std::invalid_argument("empty forest");
    }

    ordered_forest prune_impl(node*& next_write) {
        node* r = next_write;
        next_write = next_write->next_;
        r->next_ = nullptr;
        r->parent_ = nullptr;

        ordered_forest f(get_allocator());
        f.first_ = r;

        return f;
    }

    void erase_impl(node*& next_write) {
        ordered_forest cut = prune_impl(next_write);

        node* cut_root = cut.first_;
        if (cut_root->child_) {
            splice_impl(next_write->parent_, next_write, cut_root->child_);
        }

        cut_root->child_ = nullptr;
    }

    iterator_mc<false> splice_impl(node* parent, node*& next_write, node* sp_first) {
        node* sp_last = nullptr;

        for (node* j = sp_first; j; j = j->next_) {
            j->parent_ = parent;
            sp_last = j;
        }
        sp_last->next_ = next_write;
        next_write = sp_first;

        return iterator_mc<false>{sp_last};
    }

    template <typename U, typename OtherAllocator>
    void copy_impl(const ordered_forest<U, OtherAllocator>& other) {
        auto copy_children = [&](auto& self, const auto& from, auto& to) -> void {
            sibling_iterator j;
            for (auto i = other.child_begin(from); i!=other.child_end(from); ++i) {
                // TODO: explicit `this` required for g++6; remove when g++6 deprecated.
                j = j? this->insert_after(j, *i): sibling_iterator(this->push_child(to, *i));
                self(self, i, j);
            }
        };

        sibling_iterator j;
        for (auto i = other.root_begin(); i!=other.root_end(); ++i) {
            j = j? insert_after(j, *i): sibling_iterator(push_front(*i));
            copy_children(copy_children, i, j);
        }
    }

    // Node creation, destruction:

    template <typename... Args>
    node* make_node(Args&&... args) {
        node* x = node_alloc_traits::allocate(node_alloc_, 1);
        try {
            node_alloc_traits::construct(node_alloc_, x);
        }
        catch (...) {
            node_alloc_traits::deallocate(node_alloc_, x, 1);
            throw;
        }

        x->item_ = item_alloc_traits::allocate(item_alloc_, 1);
        try {
            item_alloc_traits::construct(item_alloc_, x->item_, std::forward<Args>(args)...);
        }
        catch (...) {
            item_alloc_traits::deallocate(item_alloc_, x->item_, 1);
            throw;
        }
        return x;
    }

    void delete_node(node* n) {
        if (!n) return;

        delete_item(n->item_);
        delete_node(n->child_);
        delete_node(n->next_);

        node_alloc_traits::destroy(node_alloc_, n);
        node_alloc_traits::deallocate(node_alloc_, n, 1);
    }

    void delete_item(V* item) {
        if (!item) return;

        item_alloc_traits::destroy(item_alloc_, item);
        item_alloc_traits::deallocate(item_alloc_, item, 1);
    }
};

// Helper class for building trees from initializer_lists. Ordered forest can be
// constructed from an initializer list of builder objects; each builder object
// represents a tree, constructed from a single value, or a pair: root value
// and a sequence of builder objects represeting the children.
//
// While the builder takes the same allocator type as its associated forest, it
// will build the temporary trees with a default-constructed allocator.

template <typename V, typename Allocator>
struct ordered_forest_builder {
    template <typename X, typename std::enable_if_t<std::is_constructible<V, X&&>::value, int> = 0>
    ordered_forest_builder(X&& x) {
        f_.emplace_front(x);
    }

    template <typename X, typename std::enable_if_t<std::is_constructible<V, X&&>::value, int> = 0>
    ordered_forest_builder(X&& x, std::initializer_list<ordered_forest_builder<V, Allocator>> children) {
        using sibling_iterator = typename ordered_forest<V, Allocator>::sibling_iterator;

        auto top = f_.emplace_front(x);
        sibling_iterator j;

        for (auto& g: children) {
            ordered_forest<V, Allocator> c(g.f_);
            j = j? f_.graft_after(j, std::move(c)): sibling_iterator(f_.graft_child(top, std::move(c)));
        }
    }

    friend class ordered_forest<V, Allocator>;

private:
    ordered_forest<V, Allocator> f_;
};

} // namespace util
} // namespace arb
