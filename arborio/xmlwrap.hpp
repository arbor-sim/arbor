#pragma once

// RAII and iterator wrappers for some libxml2 objects.

#include <any>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>

#include <arbor/util/expected.hpp>
#include <arborio/xml.hpp>

namespace arborio {
namespace xmlwrap {

struct bad_property {
    std::string error;
    unsigned line = 0;
};

// `non_negative` represents the corresponding constraint in the schema, which
// can mean any arbitrarily large non-negtative integer value.
//
// A faithful representation would use an arbitrary-size 'big' integer or
// a string, but for ease of implementation (and a bit more speed) we restrict
// it to whatever we can fit in an unsigned long long.

using non_negative = unsigned long long;

// String wrappers around `to_chars` for attribute types we care about.
// (`nl` is meant to stand for "no locale".)

std::string nl_to_string(non_negative);
std::string nl_to_string(long long);

// Parse attribute content as the representation of a specific type.
// Return true if successful.

bool nl_from_cstr(std::string& out, const char* content);
bool nl_from_cstr(non_negative& out, const char* content);
bool nl_from_cstr(long long& out, const char* content);
bool nl_from_cstr(double& out, const char* content);

// Wrap xmlChar* NUL-terminated string that requires deallocation.

struct xml_string {
    explicit xml_string(const xmlChar* p): p_(p, xml_string::deleter) {}

    operator const char*() const {
        return reinterpret_cast<const char*>(p_.get());
    }

private:
    std::shared_ptr<const xmlChar> p_;
    static void deleter(const xmlChar* x) { xmlFree((void*)x); }
};

// Wrappers below are generally constructed with two arguments,
// a pointer corresponding to the libxml2 object, and a dependency
// object (typically shared_ptr<X> for some X) that guards the
// lifetime of another object upon which this depends.

template <typename XmlType>
void trivial_dealloc(XmlType*) {}

template <typename XmlType, void (*xml_dealloc)(XmlType *) = &trivial_dealloc<XmlType>>
struct xml_base {
    xml_base(XmlType* p, std::any depends = {}):
        p_(p, xml_dealloc),
        depends_(std::move(depends))
    {}

protected:
    // Access to raw wrapped pointer type.
    XmlType* get() const { return p_.get(); }

    // Copy of shared_ptr<> governing lifetime of referenced object.
    auto self() const { return p_; }

    // Copy of dependency object.
    std::any depends() const { return depends_; }

private:
    std::shared_ptr<XmlType> p_;
    std::any depends_;
};

// xmlNode RAII wrapper (non-owning).

struct xml_node: protected xml_base<xmlNode>  {
    using base = xml_base<xmlNode>;
    explicit xml_node(xmlNode* p, std::any depends):
        base(p, std::move(depends))
    {}

    bool is_element() const { return get()->type==XML_ELEMENT_NODE; }
    bool is_attr() const { return get()->type==XML_ATTRIBUTE_NODE; }
    xml_string content() const { return xml_string(xmlNodeGetContent(get())); }
    unsigned line() const { return get()->line; }

    bool has_prop(const char* name) const { return xmlHasProp(get(), (const xmlChar*)name); }

    template <typename T>
    arb::util::expected<T, bad_property> prop(const char* name, std::optional<T> default_value = std::nullopt) const {
        using arb::util::unexpected;

        xmlChar* c = xmlGetProp(get(), (const xmlChar*)(name));
        if (!c) {
            if (default_value) return default_value.value();
            else return unexpected(bad_property{"missing required attribute", get()->line});
        }

        T v;
        if (nl_from_cstr(v, reinterpret_cast<const char*>(c))) return v;
        else return unexpected(bad_property{"attribute type error", get()->line});
    }

    using base::get; // (unsafe access)
};

// xmlNodeSet RAII wrapper; resource lifetime is governed by an xmlXPathObject.

struct xml_nodeset: protected xml_base<xmlNodeSet>  {
    using base = xml_base<xmlNodeSet>;

    xml_nodeset(): xml_nodeset(nullptr, std::any{}) {}

    xml_nodeset(xmlNodeSet* p, std::any depends):
        base(p, std::move(depends))
    {}

    struct iterator {
        using value_type = xml_node;
        using difference_type = std::ptrdiff_t;
        using reference = value_type; // yeah, not a real random access iterator
        using pointer = value_type*;
        using iterator_category = std::random_access_iterator_tag;

        explicit iterator(xmlNodePtr* p, const xml_nodeset* ns_ptr): p_(p), ns_ptr_(ns_ptr) {}

        bool operator==(iterator i) const { return p_==i.p_; }
        bool operator!=(iterator i) const { return p_!=i.p_; }
        bool operator<(iterator i) const { return p_<i.p_; }
        bool operator<=(iterator i) const { return p_<=i.p_; }
        bool operator>(iterator i) const { return p_>i.p_; }
        bool operator>=(iterator i) const { return p_>=i.p_; }

        reference operator*() const { return ns_ptr_->mk_xml_node(*p_); }

        struct ptr_proxy {
            xml_node inner_;
            const xml_node* operator->() const { return &inner_; }
        };
        ptr_proxy operator->() const { return ptr_proxy{ns_ptr_->mk_xml_node(*p_)}; }

        iterator& operator++() { return ++p_, *this; }
        iterator operator++(int) {
            iterator x(*this);
            return ++p_, x;
        }

        iterator& operator--() { return --p_, *this; }
        iterator operator--(int) {
            iterator x(*this);
            return --p_, x;
        }

        iterator& operator+=(ptrdiff_t n) { return p_ += n, *this; }
        iterator& operator-=(ptrdiff_t n) { return p_ -= n, *this; }
        reference operator[](ptrdiff_t n) { return *(*this+n); }

        iterator operator+(ptrdiff_t n) {
            iterator i(*this);
            return i += n;
        }
        friend iterator operator+(ptrdiff_t n, iterator i) { return i+n; }

        iterator operator-(ptrdiff_t n) {
            iterator i(*this);
            return i -= n;
        }
        friend iterator operator-(ptrdiff_t n, iterator i) { return i-n; }

        ptrdiff_t operator=(iterator i) { return p_-i.p_; }

    private:
        xmlNode** p_;
        const xml_nodeset* ns_ptr_;
    };

    iterator begin() const { return iterator{get()? get()->nodeTab: nullptr, this}; }
    iterator end() const { return iterator{get()? get()->nodeTab+get()->nodeNr: nullptr, this}; }

    iterator::reference operator[](int i) const { return begin()[i]; }
    std::size_t size() const { return get()? get()->nodeNr: 0u; }
    bool empty() const { return size()==0u; }

private:
    // Construct xml_node wrapper with the same lifetime dependency as this node set.
    xml_node mk_xml_node(xmlNode* p) const {
        return xml_node{p, depends()};
    }
};

// xmlPathObj RAII wrapper; lifetime of xmlPathObj governs lifetime of node set.

struct xml_xpathobj: protected xml_base<xmlXPathObject, xmlXPathFreeObject> {
    using base = xml_base<xmlXPathObject, xmlXPathFreeObject>;

    explicit xml_xpathobj(xmlXPathObject* p, std::any depends):
        base(p, std::move(depends))
    {}

    xml_nodeset nodes() {
        return get()->type==XPATH_NODESET? xml_nodeset{get()->nodesetval, self()}: xml_nodeset{};
    }
};

// xmlXPathContext RAII wrapper.

struct xml_xpathctx: protected xml_base<xmlXPathContext, xmlXPathFreeContext> {
    using base = xml_base<xmlXPathContext, xmlXPathFreeContext>;

    explicit xml_xpathctx(xmlXPathContext* p, std::any depends):
        base(p, std::move(depends))
    {}

    void register_ns(const char* ns, const char* uri) {
        xmlXPathRegisterNs(get(), (const xmlChar*)ns, (const xmlChar*)uri);
    }

    xml_nodeset query(const char* q) {
        return xml_xpathobj{xmlXPathEvalExpression((xmlChar*)q, get()), self()}.nodes();
    }
    xml_nodeset query(const std::string& q) { return query(q.c_str()); }

    xml_nodeset query(xml_node context, const char* q) {
        return xml_xpathobj{xmlXPathNodeEval(context.get(), (xmlChar*)q, get()), self()}.nodes();
    }
    xml_nodeset query(xml_node context, const std::string& q) { return query(std::move(context), q.c_str()); }
};

// xmlDoc RAII wrapper.

struct xml_doc: protected xml_base<xmlDoc, xmlFreeDoc> {
    using base = xml_base<xmlDoc, xmlFreeDoc>;

    xml_doc(): xml_doc(nullptr) {}

    explicit xml_doc(std::string the_doc):
        // 'Pretty sure' we don't need to keep the string after the tree is built. Pretty sure.
        xml_doc(xmlReadMemory(the_doc.c_str(), the_doc.length(), "", nullptr, xml_options))
    {}

    // TODO: (... add other ctors ...)

    friend xml_xpathctx xpath_context(const xml_doc& doc) {
        return xml_xpathctx{xmlXPathNewContext(doc.get()), doc.self()};
    }

    explicit operator bool() const { return get(); }

private:
    explicit xml_doc(xmlDoc* p): base(p) {}
    static constexpr int xml_options = XML_PARSE_NOENT | XML_PARSE_NONET;
};

// Escape a string for use as string expression within an XPath expression.

inline std::string xpath_escape(const std::string& x) {
    auto npos = std::string::npos;
     if (x.find_first_of("'")==npos) {
         return "'"+x+"'";
     }
     else if (x.find_first_of("\"")==npos) {
         return "\""+x+"\"";
     }
     else {
         std::string r = "concat(";
         std::string::size_type i = 0;
         for (;;) {
             auto j = x.find_first_of("'", i);
             r += "'";
             r.append(x, i, j==npos? j: j-i);
             r += "'";
             if (j==npos) break;
             r += ",\"";
             i = j+1;
             j = x.find_first_not_of("'",i);
             r.append(x, i, j==npos? j: j-i);
             r += "\"";
             if (j==npos) break;
             r += ",";
             i = j+1;
         }
         r += ")";
         return r;
     }
}

// Error management:
//
// Use xml_error_scope to catch libxml2 warnings and errors. The
// xml_error_scope object will restore the original error handling
// behaviour on destruction.
//
// Errors are turned into arborio::xml_error exceptions and thrown,
// while warnings are ignored (libxml2 warnings are highly innocuous).

struct xml_error_scope {
    xml_error_scope();
    ~xml_error_scope();

    xmlGenericErrorFunc generic_handler_;
    void* generic_context_;

    xmlStructuredErrorFunc structured_handler_;
    void* structured_context_;
};

} // namespace xmlwrap
} // namespace arborio
