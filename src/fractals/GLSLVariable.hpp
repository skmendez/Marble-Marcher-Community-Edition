//
// Created by Sebastian on 12/7/2020.
//

#ifndef GLSLVARIABLE_HPP_
#define GLSLVARIABLE_HPP_

#if !defined(__gl_h_) && !defined(__GL_H__) && !defined(_GL_H) && !defined(__X_GL_H)
#include <GL/glew.h>
#endif

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

template<typename T>
class GLSLVariable {
 public:
  [[nodiscard]] virtual std::string GetGLSLVariable() const = 0;
  [[nodiscard]] virtual T GetVar() const = 0;
  virtual void UpdateUniform(GLuint ProgramID) const = 0;
};

template<typename T>
class GLSLConstant : public GLSLVariable<T> {
 public:
  explicit GLSLConstant(T var) : var_(var) {}

  [[nodiscard]] std::string GetGLSLVariable() const override;

  [[nodiscard]] T GetVar() const override {
    return var_;
  }

  void UpdateUniform(GLuint ProgramID) const override {/* no uniform to update */};

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

template<>
inline std::string GLSLConstant<Eigen::Vector3f>::GetGLSLVariable() const {
  return GetMatrix("vec3");
}

template<>
inline std::string GLSLConstant<Eigen::Vector2f>::GetGLSLVariable() const {
  return GetMatrix("vec2");
}

template<>
inline std::string GLSLConstant<Eigen::Matrix3f>::GetGLSLVariable() const {
  return GetMatrix("mat3");
}

template<>
inline std::string GLSLConstant<Eigen::Matrix2f>::GetGLSLVariable() const {
  return GetMatrix("mat2");
}

template<>
inline std::string GLSLConstant<float>::GetGLSLVariable() const {
  std::stringstream ss;
  ss << std::showpoint << var_;
  return ss.str();
}

template<>
inline std::string GLSLConstant<int>::GetGLSLVariable() const {
  return std::to_string(var_);
}


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

  void UpdateUniform(GLuint ProgramID) const override {
    glUseProgram(ProgramID);
    SetUniformFromLoc(glGetUniformLocation(ProgramID, name_.c_str()));
  }

 private:
  void SetUniformFromLoc(GLuint A) const;
  T var_;
  const std::string name_;
};

template<>
inline void GLSLUniform<Eigen::Vector3f>::SetUniformFromLoc(GLuint A) const {
  glUniform3fv(A, 1, var_.data());
}

template<>
inline void GLSLUniform<Eigen::Vector2f>::SetUniformFromLoc(GLuint A) const {
  glUniform2fv(A, 1, var_.data());
}

template<>
inline void GLSLUniform<Eigen::Matrix3f>::SetUniformFromLoc(GLuint A) const {
  glUniformMatrix3fv(A, 1, true, var_.data());
}

template<>
inline void GLSLUniform<Eigen::Matrix2f>::SetUniformFromLoc(GLuint A) const {
  glUniformMatrix2fv(A, 1, true, var_.data());
}

template<>
inline void GLSLUniform<float>::SetUniformFromLoc(GLuint A) const {
  glUniform1f(A, var_);
}

template<>
inline void GLSLUniform<int>::SetUniformFromLoc(GLuint A) const {
  glUniform1i(A, var_);
}

#endif //GLSLVARIABLE_HPP_
