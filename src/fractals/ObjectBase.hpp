//
// Created by Sebastian on 12/1/2020.
//

#ifndef OBJECTBASE_HPP_
#define OBJECTBASE_HPP_
#include <Eigen/Dense>

class ObjectBase {
 public:
  ObjectBase() = default;
  virtual float DistanceEstimator(Eigen::Vector4f p) = 0;
  virtual Eigen::Vector3f NearestPoint(Eigen::Vector4f p) = 0;
};


#endif //OBJECTBASE_HPP_
