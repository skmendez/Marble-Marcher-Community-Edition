//
// Created by Sebastian on 12/7/2020.
//

#ifndef GLSLVARIABLE_HPP_
#define GLSLVARIABLE_HPP_

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

template<typename T>
class GLSLVariable {
 public:
  [[nodiscard]] virtual std::string GetGLSLVariable() const = 0;
  [[nodiscard]] virtual T GetVar() const = 0;
};

template<typename T>
class GLSLConstant : public GLSLVariable<T> {
 public:
  explicit GLSLConstant(T var) : var_(var) {}

  [[nodiscard]] std::string GetGLSLVariable() const override;

  [[nodiscard]] T GetVar() const override {
    return var_;
  }

 private:
  const T var_;
  [[nodiscard]] std::string GetMatrix(const std::string& prefix) const {
    std::stringstream ss;
    ss << std::showpoint;
    Eigen::IOFormat genericFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", prefix + "(", ")");
    ss << var_.format(genericFormat);
    return ss.str();
  }
};





template <typename T>
class GLSLUniform : public GLSLVariable<T> {
 public:
  GLSLUniform(T var, std::string name) : var_(var), name_(std::move(name)) {}

  [[nodiscard]] std::string GetGLSLVariable() const override  {
    return name_;
  }

  [[nodiscard]] std::string GetName() const {
    return name_;
  }

  [[nodiscard]] T GetVar() const override {
    return var_;
  }

  void SetVar(T v) {
    var_ = v;
  }

  void SetUniform(unsigned int ProgramID) {
    SetUniformFromLoc(glGetUniformLocation(ProgramID, name_.c_str()));
  }

 private:
  void SetUniformFromLoc(unsigned int A) {};
  T var_;
  const std::string name_;
};


#endif //GLSLVARIABLE_HPP_
