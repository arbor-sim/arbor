#include "textbuffer.hpp"

/******************************************************************************
                              TextBuffer
******************************************************************************/
TextBuffer& TextBuffer::add_gutter() {
    text_ << gutter_;
    return *this;
}
void TextBuffer::add_line(std::string const& line) {
    text_ << gutter_ << line << std::endl;
}
void TextBuffer::add_line() {
    text_ << std::endl;
}
void TextBuffer::end_line(std::string const& line) {
    text_ << line << std::endl;
}
void TextBuffer::end_line() {
    text_ << std::endl;
}

std::string TextBuffer::str() const {
    return text_.str();
}

void TextBuffer::set_gutter(int width) {
    indent_ = width;
    gutter_ = std::string(indent_, ' ');
}

void TextBuffer::increase_indentation() {
    indent_ += indentation_width_;
    if(indent_<0) {
        indent_=0;
    }
    gutter_ = std::string(indent_, ' ');
}
void TextBuffer::decrease_indentation() {
    indent_ -= indentation_width_;
    if(indent_<0) {
        indent_=0;
    }
    gutter_ = std::string(indent_, ' ');
}

std::stringstream& TextBuffer::text() {
    return text_;
}
