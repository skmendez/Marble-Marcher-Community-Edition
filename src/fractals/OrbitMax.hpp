//
// Created by Sebastian on 12/8/2020.
//

#ifndef ORBITMAX_HPP_
#define ORBITMAX_HPP_

#include <utility>

#include "OrbitBase.hpp"

class OrbitMax : public OrbitBase {
 public:
  OrbitMax(std::shared_ptr<GLSLVariable<Eigen::Vector3f>>  frac_color) : frac_color_(std::move(frac_color)) {}

  void UpdateUniforms(unsigned int ProgramID) const override {
    frac_color_->UpdateUniform(ProgramID);
  }

 protected:
  void GLSLIfColor(GLSLFractalCode& buf) const override {
    buf << "orbit = max(orbit, p.xyz * " <<  frac_color_->GetGLSLVariable() << ");\n";
  }

 private:
  std::shared_ptr<GLSLVariable<Eigen::Vector3f>> frac_color_;
};


#endif //ORBITMAX_HPP_
