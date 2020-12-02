//
// Created by Sebastian on 12/2/2020.
//

#ifndef GLSLBASE_HPP_
#define GLSLBASE_HPP_

#include <assert.h>
#include <ostream>
#include <iostream>

class IndentableOStreamBuf : private std::streambuf, public std::ostream {
 public:
  explicit IndentableOStreamBuf(std::ostream& dest) : std::ostream(this), dest_(dest) {}
  void IncreaseIndent() {
    myIndent = std::string(myIndent.size() + indent_amount_, ' ');
  }

  void DecreaseIndent() {
    assert(myIndent.size() >= indent_amount_);
    myIndent = std::string(myIndent.size() - indent_amount_, ' ');
  }

 protected:
  virtual int overflow(int ch) {
    if (myIsAtStartOfLine && ch != '\n') {
      dest_.write(myIndent.data(), myIndent.size());
    }
    myIsAtStartOfLine = ch == '\n';
    dest_.put(ch);
    return ch;
  }


 private:
  std::ostream& dest_;
  bool myIsAtStartOfLine = true;
  std::string myIndent = std::string();
static const int indent_amount_ = 4;
};


class GLSLBase {
 public:
  virtual void GLSL(IndentableOStreamBuf& buf) = 0;
};


#endif //GLSLBASE_HPP_
