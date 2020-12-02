//
// Created by Sebastian on 12/2/2020.
//

#ifndef FOLDROTATE_HPP_
#define FOLDROTATE_HPP_

enum Axis {X = 0, Y = 1, Z = 2};

#include "FoldableBase.hpp"

class FoldRotate : public FoldableBase {
 public:
  FoldRotate(Axis rotation_axis, float radians) : rotation_axis_(rotation_axis), radians_(radians) {}

  void Fold(Eigen::Vector4f& p) override {
    p.segment<3>(0) = Rotate(radians_, p.segment<3>(0));
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) override {
    n = Rotate(-radians_, n);
  }

  void GLSL(IndentableOStreamBuf& buf) override {
    char rotation_char = 'X' + rotation_axis_;
    switch (rotation_axis_) {
      case Axis::X:
        break;
      case Axis::Y:
        break;
      case Axis::Z:
        break;
    }
  }
 private:
  Eigen::Vector3f Rotate(float radians, Eigen::Vector3f p) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    // We'll assume that we are rotating on the X plane by default;
    // AccessComponent will shift based on the actual axis of rotation.
    Axis component1 = AccessComponent(Axis::Y);
    Axis component2 = AccessComponent(Axis::Z);

    const float component1_rot = c*p(component1) + s*p(component2);
    const float component2_rot = c*p(component2) - s*p(component1);
    p(component1) = component1_rot;
    p(component2) = component2_rot;
    return p;
  }

  Axis AccessComponent(Axis orig) {
    return static_cast<Axis>((orig + rotation_axis_) % 3);
  }

  Axis rotation_axis_;

 private:
  float radians_;
};


#endif //FOLDROTATE_HPP_
