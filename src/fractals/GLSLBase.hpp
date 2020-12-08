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

  void SetPassType(bool is_color) {
    is_color_ = is_color;
  }

  bool isColorPass() const {
    return is_color_;
  }

 protected:
  int overflow(int ch) override {
    if (start_of_line && ch != '\n') {
      for (int i = 0; i < indent_length; i++) {
        ss_.put(indent_char_);
      }
    }
    start_of_line = ch == '\n';
    ss_.put(ch);
    return ch;
  }

 private:
  bool is_color_ = false;
  std::stringstream ss_{};
  bool start_of_line = true;
  int indent_length = 0;
  static const int indent_amount_ = 1;
  static const char indent_char_ = '\t';
};


class GLSLBase {
 public:
  virtual void GLSL(GLSLFractalCode& buf) const = 0;
  virtual void UpdateUniforms(unsigned int ProgramID) const = 0;
};


#endif //GLSLBASE_HPP_
