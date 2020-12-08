//
// Created by Sebastian on 12/2/2020.
//

#ifndef FOLDROTATE_HPP_
#define FOLDROTATE_HPP_

#include <utility>

#include "FoldableBase.hpp"

class FoldRotate : public FoldableBase {
 public:
  FoldRotate(Axis rotation_axis,
      std::shared_ptr<GLSLVariable<Eigen::Matrix2f>> rot_mat) :
  rotation_axis_(rotation_axis), rot_mat_(std::move(rot_mat)) {}

  void Fold(Eigen::Vector4f& p) const override {
    p.segment<3>(0) = Rotate(false, p.segment<3>(0));
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const override {
    n = Rotate(true, n);
  }

  void GLSL(GLSLFractalCode& buf) const override {
    std::string vars;
    switch (rotation_axis_) {
      case Axis::X:
        vars = "yz";
        break;
      case Axis::Y:
        vars = "xz";
        break;
      case Axis::Z:
        vars = "xy";
        break;
    }

    buf << "p." << vars << " *= " << rot_mat_->GetGLSLVariable() << ";\n";
  }

  void UpdateUniforms(unsigned int ProgramID) const override {
    rot_mat_->UpdateUniform(ProgramID);
  }

 private:
  Eigen::Vector3f Rotate(bool unrotate, Eigen::Vector3f p) const {
    // We'll assume that we are rotating on the X plane by default;
    // AccessComponent will shift based on the actual axis of rotation.
    Axis component1 = AccessComponent(Axis::Y);
    Axis component2 = AccessComponent(Axis::Z);
    Eigen::Vector2f two_vec{p(component1), p(component2)};
    if (unrotate) {
      two_vec = rot_mat_->GetVar().transpose() * two_vec;
    } else {
      two_vec = rot_mat_->GetVar() * two_vec;
    }
    p(component1) = two_vec(0);
    p(component2) = two_vec(1);
    return p;
  }

  Axis AccessComponent(Axis orig) const {
    return static_cast<Axis>((orig + rotation_axis_) % 3);
  }

  const Axis rotation_axis_;
  std::shared_ptr<GLSLVariable<Eigen::Matrix2f>> rot_mat_;
};


#endif //FOLDROTATE_HPP_
