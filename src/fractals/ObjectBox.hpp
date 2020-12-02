//
// Created by Sebastian on 12/1/2020.
//

#ifndef OBJECTBOX_HPP_
#define OBJECTBOX_HPP_


#include <utility>

#include "ObjectBase.hpp"

class ObjectBox : public ObjectBase {
 public:
  explicit ObjectBox(Eigen::Vector3f box_shape) : box_shape_(std::move(box_shape)) {}

  float DistanceEstimator(Eigen::Vector4f p) override {
    const Eigen::Vector3f a = p.segment<3>(0).cwiseAbs() - box_shape_;
    return (std::min(std::max(std::max(a.x(), a.y()), a.z()), 0.0f) + a.cwiseMax(0.0f).norm()) / p.w();
  }

  Eigen::Vector3f NearestPoint(Eigen::Vector4f p) override {
    return p.segment<3>(0).cwiseMax(-box_shape_).cwiseMin(box_shape_);
  }

 private:
  Eigen::Vector3f box_shape_;
};


#endif //OBJECTBOX_HPP_
