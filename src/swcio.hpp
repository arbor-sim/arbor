#pragma once

#include <exception>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace nest {
namespace mc {
namespace io {

class swc_record
{
public:
    using id_type = int;

    // FIXME: enum's are not completely type-safe, since they can accept
    // anything that can be casted to their underlying type.
    //
    // More on SWC files: http://research.mssm.edu/cnic/swc.html
    enum kind {
        undefined = 0,
        soma,
        axon,
        dendrite,
        apical_dendrite,
        fork_point,
        end_point,
        custom
    };

    // swc records assume zero-based indexing; root's parent remains -1
    swc_record(kind type, int id,
                float x, float y, float z, float r,
                int parent_id)
        : type_(type)
        , id_(id)
        , x_(x)
        , y_(y)
        , z_(z)
        , r_(r)
        , parent_id_(parent_id)
    {
        check_consistency();
    }

    swc_record()
        : type_(swc_record::undefined)
        , id_(0)
        , x_(0)
        , y_(0)
        , z_(0)
        , r_(0)
        , parent_id_(-1)
    { }

    swc_record(const swc_record &other) = default;
    swc_record &operator=(const swc_record &other) = default;

    bool strict_equals(const swc_record &other) const
    {
        return id_ == other.id_ &&
            x_ == other.x_ &&
            y_ == other.y_ &&
            z_ == other.z_ &&
            r_ == other.r_ &&
            parent_id_ == other.parent_id_;
    }

    // Equality and comparison operators
    friend bool operator==(const swc_record &lhs,
                           const swc_record &rhs)
    {
        return lhs.id_ == rhs.id_;
    }

    friend bool operator<(const swc_record &lhs,
                          const swc_record &rhs)
    {
        return lhs.id_ < rhs.id_;
    }

    friend bool operator<=(const swc_record &lhs,
                           const swc_record &rhs)
    {
        return (lhs < rhs) || (lhs == rhs);
    }

    friend bool operator!=(const swc_record &lhs,
                           const swc_record &rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator>(const swc_record &lhs,
                          const swc_record &rhs)
    {
        return !(lhs < rhs) && (lhs != rhs);
    }

    friend bool operator>=(const swc_record &lhs,
                           const swc_record &rhs)
    {
        return !(lhs < rhs);
    }

    friend std::ostream &operator<<(std::ostream &os, const swc_record &record);

    kind type() const
    {
        return type_;
    }

    id_type id() const
    {
        return id_;
    }

    id_type parent() const
    {
        return parent_id_;
    }

    float x() const
    {
        return x_;
    }

    float y() const
    {
        return y_;
    }

    float z() const
    {
        return z_;
    }

    float radius() const
    {
        return r_;
    }

    float diameter() const
    {
        return 2*r_;
    }

    void renumber(id_type new_id, std::map<id_type, id_type> &idmap);

private:
    void check_consistency() const;

    kind type_;         // record type
    id_type id_;        // record id
    float x_, y_, z_;   // record coordinates
    float r_;           // record radius
    id_type parent_id_; // record parent's id
};


class swc_parse_error : public std::runtime_error
{
public:
    explicit swc_parse_error(const char *msg, std::size_t lineno)
        : std::runtime_error(msg)
        , lineno_(lineno)
    { }

    explicit swc_parse_error(const std::string &msg, std::size_t lineno)
        : std::runtime_error(msg)
        , lineno_(lineno)
    { }

    std::size_t lineno() const
    {
        return lineno_;
    }

private:
    std::size_t lineno_;
};

class swc_parser
{
public:
    swc_parser(const std::string &delim,
               std::string comment_prefix)
        : delim_(delim)
        , comment_prefix_(comment_prefix)
        , lineno_(0)
    { }

    swc_parser()
        : delim_(" ")
        , comment_prefix_("#")
        , lineno_(0)
    { }

    std::size_t lineno() const
    {
        return lineno_;
    }

    std::istream &parse_record(std::istream &is, swc_record &record);

private:
    // Read the record from a string stream; will be treated like a single line
    swc_record parse_record(std::istringstream &is);

    std::string delim_;
    std::string comment_prefix_;
    std::string linebuff_;
    std::size_t lineno_;
};


std::istream &operator>>(std::istream &is, swc_record &record);

class swc_record_stream_iterator :
        public std::iterator<std::forward_iterator_tag, swc_record>
{
public:
    struct eof_tag { };

    swc_record_stream_iterator(std::istream &is)
        : is_(is)
        , eof_(false)
    {
        is_.clear();
        is_.seekg(std::ios_base::beg);
        read_next_record();
    }

    swc_record_stream_iterator(std::istream &is, eof_tag)
        : is_(is)
        , eof_(true)
    { }

    swc_record_stream_iterator(const swc_record_stream_iterator &other)
        : is_(other.is_)
        , parser_(other.parser_)
        , curr_record_(other.curr_record_)
        , eof_(other.eof_)
    { }

    swc_record_stream_iterator &operator++()
    {
        if (eof_) {
            throw std::out_of_range("attempt to read past eof");
        }

        read_next_record();
        return *this;
    }

    swc_record_stream_iterator operator++(int)
    {
        swc_record_stream_iterator ret(*this);
        operator++();
        return ret;
    }

    value_type operator*() const
    {
        if (eof_) {
            throw std::out_of_range("attempt to read past eof");
        }

        return curr_record_;
    }

    bool operator==(const swc_record_stream_iterator &other) const
    {
        if (eof_ && other.eof_) {
            return true;
        } else {
            return curr_record_.strict_equals(other.curr_record_);
        }
    }

    bool operator!=(const swc_record_stream_iterator &other) const
    {
        return !(*this == other);
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const swc_record_stream_iterator &iter)
    {
        os << "{ is_.tellg(): " << iter.is_.tellg()  << ", "
           << "curr_record_: "  << iter.curr_record_ << ", "
           << "eof_: "          << iter.eof_         << "}";

        return os;
    }

private:
    void read_next_record()
    {
        parser_.parse_record(is_, curr_record_);
        if (is_.eof()) {
            eof_ = true;
        }
    }

    std::istream &is_;
    swc_parser parser_;
    swc_record curr_record_;

    // indicator of eof; we need a way to define an end() iterator without
    // seeking to the end of file
    bool eof_;
};


class swc_record_range_raw
{
public:
    using value_type      = swc_record;
    using reference       = value_type &;
    using const_reference = const value_type &;
    using iterator        = swc_record_stream_iterator;
    using const_iterator  = const swc_record_stream_iterator;

    swc_record_range_raw(std::istream &is)
        : is_(is)
    { }

    iterator begin() const
    {
        return swc_record_stream_iterator(is_);
    }

    iterator end() const
    {
        iterator::eof_tag eof;
        return swc_record_stream_iterator(is_, eof);
    }

    bool empty() const
    {
        return begin() == end();
    }

private:
    std::istream &is_;
};

//
// Reads records from an input stream until an eof is encountered and returns a
// cleaned sequence of swc records.
//
// For more information check here:
//   https://github.com/eth-cscs/cell_algorithms/wiki/SWC-file-parsing
//

class swc_record_range_clean
{
public:
    using value_type     = swc_record;
    using reference      = value_type &;
    using const_referene = const value_type &;
    using iterator       = std::vector<swc_record>::iterator;
    using const_iterator = std::vector<swc_record>::const_iterator;

    swc_record_range_clean(std::istream &is);

    iterator begin()
    {
        return records_.begin();
    }

    iterator end()
    {
        return records_.end();
    }

    std::size_t size()
    {
        return records_.size();
    }

    bool empty() const
    {
        return records_.empty();
    }

private:
    std::vector<swc_record> records_;
};

struct swc_io_raw
{
    using record_range_type = swc_record_range_raw;
};

struct swc_io_clean
{
    using record_range_type = swc_record_range_clean;
};

template<typename T = swc_io_clean>
 typename T::record_range_type swc_get_records(std::istream &is)
{
    return typename T::record_range_type(is);
}

} // namespace io
} // namespace mc
} // namespace nest

