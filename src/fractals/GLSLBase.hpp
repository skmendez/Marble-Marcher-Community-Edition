//
// Created by Sebastian on 12/2/2020.
//

#ifndef GLSLBASE_HPP_
#define GLSLBASE_HPP_

#include <assert.h>
#include <ostream>
#include <iostream>

class GLSLFractalCode : private std::streambuf, public std::ostream {
 public:
  explicit GLSLFractalCode() : std::ostream(this) {}
  void IncreaseIndent() {
    indent_length += indent_amount_;
  }

  void DecreaseIndent() {
    assert(indent_length >= indent_amount_);
    indent_length = std::max(indent_length - indent_amount_, 0);
  }

  [[nodiscard]] std::string get() const {
    return ss_.str();
  }

 protected:
  int overflow(int ch) override {
    if (start_of_line && ch != '\n') {
      for (int i = 0; i < indent_length; i++) {
        ss_.put(' ');
      }
    }
    start_of_line = ch == '\n';
    ss_.put(ch);
    return ch;
  }

 private:
  std::stringstream ss_{};
  bool start_of_line = true;
  int indent_length = 0;
  static const int indent_amount_ = 4;
};


class GLSLBase {
 public:
  virtual void GLSL(GLSLFractalCode& buf) const = 0;
};


#endif //GLSLBASE_HPP_
