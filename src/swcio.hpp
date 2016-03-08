#pragma once

#include <exception>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace nestmc
{

namespace io
{


class cell_record 
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

    // cell records assume zero-based indexing; root's parent remains -1
    cell_record(kind type, int id, 
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
    
    cell_record()
        : type_(cell_record::undefined)
        , id_(0)
        , x_(0)
        , y_(0)
        , z_(0)
        , r_(0)
        , parent_id_(-1)
    { }

    cell_record(const cell_record &other) = default;
    cell_record &operator=(const cell_record &other) = default;

    bool strict_equals(const cell_record &other)
    {
        return id_ == other.id_ &&
            x_ == other.x_ &&
            y_ == other.y_ &&
            z_ == other.z_ &&
            r_ == other.r_ &&
            parent_id_ == other.parent_id_;
    }

    // Equality and comparison operators
    friend bool operator==(const cell_record &lhs,
                           const cell_record &rhs)
    {
        return lhs.id_ == rhs.id_;
    }

    friend bool operator<(const cell_record &lhs,
                          const cell_record &rhs)
    {
        return lhs.id_ < rhs.id_;
    }

    friend bool operator<=(const cell_record &lhs,
                           const cell_record &rhs)
    {
        return (lhs < rhs) || (lhs == rhs);
    }

    friend bool operator!=(const cell_record &lhs,
                           const cell_record &rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator>(const cell_record &lhs,
                          const cell_record &rhs)
    {
        return !(lhs < rhs) && (lhs != rhs);
    }

    friend bool operator>=(const cell_record &lhs,
                           const cell_record &rhs)
    {
        return !(lhs < rhs);
    }

    friend std::ostream &operator<<(std::ostream &os, const cell_record &cell);

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

    kind type_;         // cell type
    id_type id_;        // cell id
    float x_, y_, z_;   // cell coordinates
    float r_;           // cell radius
    id_type parent_id_; // cell parent's id
};


class swc_parse_error : public std::runtime_error
{
public:
    explicit swc_parse_error(const char *msg)
        : std::runtime_error(msg)
    { }

    explicit swc_parse_error(const std::string &msg)
        : std::runtime_error(msg)
    { }
};

class swc_parser
{
public:
    swc_parser(const std::string &delim,
               std::string comment_prefix)
        : delim_(delim)
        , comment_prefix_(comment_prefix)
    { }

    swc_parser()
        : delim_(" ")
        , comment_prefix_("#")
    { }

    std::istream &parse_record(std::istream &is, cell_record &cell);

private:
    // Read the record from a string stream; will be treated like a single line
    cell_record parse_record(std::istringstream &is);

    std::string delim_;
    std::string comment_prefix_;
    std::string linebuff_;
};


std::istream &operator>>(std::istream &is, cell_record &cell);

//
// Reads cells from an input stream until an eof is encountered and returns a
// cleaned sequence of cell records.
//
// For more information check here:
//   https://github.com/eth-cscs/cell_algorithms/wiki/SWC-file-parsing
//
std::vector<cell_record> swc_read_cells(std::istream &is);

class cell_record_stream_iterator :
        public std::iterator<std::forward_iterator_tag, cell_record>
{
public:
    struct eof_tag { };

    cell_record_stream_iterator(std::istream &is)
        : is_(is)
        , eof_(false)
    {
        read_next_record();
    }

    cell_record_stream_iterator(std::istream &is, eof_tag)
        : is_(is)
        , eof_(true)
    { }


    cell_record_stream_iterator &operator++()
    {
        if (eof_) {
            throw std::out_of_range("attempt to read past eof");
        }

        read_next_record();
        return *this;
    }

    cell_record_stream_iterator operator++(int);

    value_type operator*()
    {
        return curr_record_;
    }

    bool operator==(const cell_record_stream_iterator &other)
    {
        if (eof_ && other.eof_) {
            return true;
        } else {
            return curr_record_.strict_equals(other.curr_record_);
        }
    }

    bool operator!=(const cell_record_stream_iterator &other)
    {
        return !(*this == other);
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
    cell_record curr_record_;

    // indicator of eof; we need a way to define an end() iterator without
    // seeking to the end of file
    bool eof_;
};


class cell_record_range_raw
{
public:
    using value_type     = cell_record;
    using reference      = value_type &;
    using const_referene = const value_type &;
    using iterator       = cell_record_stream_iterator;
    using const_iterator = const cell_record_stream_iterator;

    cell_record_range_raw(std::istream &is)
        : is_(is)
    { }

    iterator begin()
    {
        return cell_record_stream_iterator(is_);
    }

    iterator end()
    {
        iterator::eof_tag eof;
        return cell_record_stream_iterator(is_, eof);
    }

private:
    std::istream &is_;
};


}   // end of nestmc::io
}   // end of nestmc
