//
// Created by Sebastian on 12/8/2020.
//

#include "GLSLVariable.hpp"

#if !defined(__gl_h_) && !defined(__GL_H__) && !defined(_GL_H) && !defined(__X_GL_H)
#include <GL/glew.h>
#endif

template<>
std::string GLSLConstant<Eigen::Vector3f>::GetGLSLVariable() const {
  return GetMatrix("vec3");
}

template<>
std::string GLSLConstant<Eigen::Vector2f>::GetGLSLVariable() const {
  return GetMatrix("vec2");
}

template<>
std::string GLSLConstant<Eigen::Matrix3f>::GetGLSLVariable() const {
  return GetMatrix("mat3");
}

template<>
std::string GLSLConstant<Eigen::Matrix2f>::GetGLSLVariable() const {
  return GetMatrix("mat2");
}

template<>
std::string GLSLConstant<float>::GetGLSLVariable() const {
  std::stringstream ss;
  ss << std::showpoint << var_;
  return ss.str();
}

template<>
std::string GLSLConstant<int>::GetGLSLVariable() const {
  return std::to_string(var_);
}

template<>
void GLSLUniform<Eigen::Vector3f>::SetUniformFromLoc(GLuint A) {
  glUniform3fv(A, 1, var_.data());
}

template<>
void GLSLUniform<Eigen::Vector2f>::SetUniformFromLoc(GLuint A) {
  glUniform2fv(A, 1, var_.data());
}

template<>
void GLSLUniform<Eigen::Matrix3f>::SetUniformFromLoc(GLuint A) {
  glUniformMatrix3fv(A, 1, true, var_.data());
}

template<>
void GLSLUniform<Eigen::Matrix2f>::SetUniformFromLoc(GLuint A) {
  glUniformMatrix2fv(A, 1, true, var_.data());
}

template<>
void GLSLUniform<float>::SetUniformFromLoc(GLuint A) {
  glUniform1f(A, var_);
}

template<>
void GLSLUniform<int>::SetUniformFromLoc(GLuint A) {
  glUniform1i(A, var_);
}