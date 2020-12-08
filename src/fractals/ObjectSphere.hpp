//
// Created by Sebastian on 12/8/2020.
//

#ifndef OBJECTSPHERE_HPP_
#define OBJECTSPHERE_HPP_

#include "ObjectBase.hpp"

class ObjectSphere : public ObjectBase {
 public:
  explicit ObjectSphere(std::shared_ptr<GLSLVariable<float>> sphere_radius) :
      sphere_radius_(std::move(sphere_radius)) {}

  float DistanceEstimator(Eigen::Vector4f p) const override {
    return (p.segment<3>(0).norm() - sphere_radius_->GetVar()) / p.w();
  }

  Eigen::Vector3f NearestPoint(Eigen::Vector4f p) const override {
    return (p.segment<3>(0).normalized()) * sphere_radius_->GetVar();
  }

  void GLSL(GLSLFractalCode& buf) const override {
    buf << "d = de_sphere(p, " << sphere_radius_->GetGLSLVariable() << ");" << std::endl;
  }

  void UpdateUniforms(unsigned int ProgramID) const override {
    sphere_radius_->UpdateUniform(ProgramID);
  }

 private:
  std::shared_ptr<GLSLVariable<float>> sphere_radius_;
};


#endif //OBJECTSPHERE_HPP_
