//
// Created by Sebastian on 12/1/2020.
//

#ifndef OBJECTBASE_HPP_
#define OBJECTBASE_HPP_
#include <Eigen/Dense>
#include "GLSLBase.hpp"
#include "GLSLVariable.hpp"

class ObjectBase : public GLSLBase {
 public:
  ObjectBase() = default;
  [[nodiscard]] virtual float DistanceEstimator(Eigen::Vector4f p) const = 0;
  [[nodiscard]] virtual Eigen::Vector3f NearestPoint(Eigen::Vector4f p) const = 0;
  void GLSL(GLSLFractalCode& buf) const override = 0;
};


#endif //OBJECTBASE_HPP_
