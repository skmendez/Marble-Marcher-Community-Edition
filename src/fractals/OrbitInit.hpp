//
// Created by Sebastian on 12/8/2020.
//

#ifndef ORBITINIT_HPP_
#define ORBITINIT_HPP_

#include <utility>

#include "OrbitBase.hpp"
#include "GLSLVariable.hpp"

class OrbitInit : public OrbitBase {
 public:
  OrbitInit(std::shared_ptr<GLSLVariable<Eigen::Vector3f>>  orbit_start) : orbit_start_(std::move(orbit_start)) {}

  void UpdateUniforms(unsigned int ProgramID) const override {
    orbit_start_->UpdateUniform(ProgramID);
  }

 protected:
  void GLSLIfColor(GLSLFractalCode& buf) const override {
    buf << "orbit = " << orbit_start_->GetGLSLVariable() << ";\n";
  }

 private:
  std::shared_ptr<GLSLVariable<Eigen::Vector3f>> orbit_start_;
};


#endif //ORBITINIT_HPP_
