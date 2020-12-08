//
// Created by Sebastian on 12/8/2020.
//

#ifndef OBJECTINTERSECT_HPP_
#define OBJECTINTERSECT_HPP_

#include "ObjectBase.hpp"

class ObjectIntersect : public ObjectBase {
 public:
  ObjectIntersect(std::unique_ptr<ObjectBase> left, std::unique_ptr<ObjectBase> right) :
      left_(std::move(left)), right_(std::move(right)) {}

  float DistanceEstimator(Eigen::Vector4f p) const override {
    return std::max(left_->DistanceEstimator(p), right_->DistanceEstimator(p));
  }

  Eigen::Vector3f NearestPoint(Eigen::Vector4f p) const override {
    float left_dist = left_->DistanceEstimator(p);
    float right_dist = right_->DistanceEstimator(p);
    if (left_dist > right_dist) {
      return left_->NearestPoint(p);
    } else {
      return right_->NearestPoint(p);
    }
  }

  void GLSL(GLSLFractalCode& buf) const override {
    buf << "vec4 original_p_inter = p;\n";
    left_->GLSL(buf);
    buf << "float old_d_inter = d;\n";
    buf << "p = original_p_inter;\n";
    if (buf.isColorPass()) {
      buf << "vec3 old_orbit_inter = orbit;\n";
    }
    right_->GLSL(buf);
    buf << "if (old_d_inter > d) { d = old_d_inter; ";
    if (buf.isColorPass()) {
      buf << " orbit = old_orbit_inter; ";
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


#endif //OBJECTINTERSECT_HPP_
