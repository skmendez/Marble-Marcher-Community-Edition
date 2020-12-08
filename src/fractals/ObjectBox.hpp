//
// Created by Sebastian on 12/1/2020.
//

#ifndef OBJECTBOX_HPP_
#define OBJECTBOX_HPP_


#include <utility>

#include "ObjectBase.hpp"

class ObjectBox : public ObjectBase {
 public:
  explicit ObjectBox(std::shared_ptr<GLSLVariable<Eigen::Vector3f>> box_shape) : box_shape_(std::move(box_shape)) {}

  float DistanceEstimator(Eigen::Vector4f p) const override {
    const Eigen::Vector3f a = p.segment<3>(0).cwiseAbs() - box_shape_->GetVar();
    return (std::min(std::max(std::max(a.x(), a.y()), a.z()), 0.0f) + a.cwiseMax(0.0f).norm()) / p.w();
  }

  Eigen::Vector3f NearestPoint(Eigen::Vector4f p) const override {
    return p.segment<3>(0).cwiseMax(-box_shape_->GetVar()).cwiseMin(box_shape_->GetVar());
  }

  void GLSL(GLSLFractalCode& buf) const override {
    buf << "d = de_box(p, " << box_shape_->GetGLSLVariable() << ");" << std::endl;
  }

 private:
  std::shared_ptr<GLSLVariable<Eigen::Vector3f>> box_shape_;
};


#endif //OBJECTBOX_HPP_
