//
// Created by Sebastian on 12/8/2020.
//

#ifndef OBJECTDIFFERENCE_HPP_
#define OBJECTDIFFERENCE_HPP_


class ObjectDifference : public ObjectBase {
 public:
  ObjectDifference(std::unique_ptr<ObjectBase> left, std::unique_ptr<ObjectBase> right) :
      left_(std::move(left)), right_(std::move(right)) {}

  float DistanceEstimator(Eigen::Vector4f p) const override {
    return std::max(left_->DistanceEstimator(p), -right_->DistanceEstimator(p));
  }

  Eigen::Vector3f NearestPoint(Eigen::Vector4f p) const override {
    float left_dist = left_->DistanceEstimator(p);
    float right_dist = -right_->DistanceEstimator(p);
    if (left_dist > right_dist) {
      return left_->NearestPoint(p);
    } else {
      return right_->NearestPoint(p);
    }
  }

  void GLSL(GLSLFractalCode& buf) const override {
    buf << "vec4 original_p_diff = p;\n";
    left_->GLSL(buf);
    buf << "float old_d_diff = d;\n";
    buf << "p = original_p_diff;\n";
    if (buf.isColorPass()) {
      buf << "vec3 old_orbit_diff = orbit;\n";
    }
    right_->GLSL(buf);
    buf << "d = -d;\n";
    buf << "if (old_d_diff > d) { d = old_d_diff; ";
    if (buf.isColorPass()) {
      buf << " orbit = old_orbit_diff; ";
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


#endif //OBJECTDIFFERENCE_HPP_
