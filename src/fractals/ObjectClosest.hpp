//
// Created by Sebastian on 12/8/2020.
//

#ifndef OBJECTCLOSEST_HPP_
#define OBJECTCLOSEST_HPP_

#include "ObjectBase.hpp"

class ObjectClosest : public ObjectBase {
 public:
  ObjectClosest(std::unique_ptr<ObjectBase> left, std::unique_ptr<ObjectBase> right) :
    left_(std::move(left)), right_(std::move(right)) {}

  float DistanceEstimator(Eigen::Vector4f p) const override {
    return std::min(left_->DistanceEstimator(p), right_->DistanceEstimator(p));
  }

  Eigen::Vector3f NearestPoint(Eigen::Vector4f p) const override {
    Eigen::Vector3f left_nearest = left_->NearestPoint(p);
    Eigen::Vector3f right_nearest = right_->NearestPoint(p);
    float left_dist = (left_nearest - p.segment<3>(0)).norm();
    float right_dist = (right_nearest - p.segment<3>(0)).norm();
    if (left_dist < right_dist) {
      return left_nearest;
    } else {
      return right_nearest;
    }
  }

  void GLSL(GLSLFractalCode& buf) const override {
    left_->GLSL(buf);
    buf << "float old_d = d;\n";
    if (buf.isColorPass()) {
      buf << "vec3 old_orbit = orbit;\n";
    }
    right_->GLSL(buf);
    buf << "if (old_d < d) { d = old_d; ";
    if (buf.isColorPass()) {
      buf << " orbit = old_orbit; ";
    }
    buf << "}\n";
  }

  void UpdateUniforms(unsigned int ProgramID) const override {
    left_->UpdateUniforms(ProgramID);
    right_->UpdateUniforms(ProgramID);
  }

 private:

  std::unique_ptr<ObjectBase> left_;
  std::unique_ptr<ObjectBase> right_;
};


#endif //OBJECTCLOSEST_HPP_
