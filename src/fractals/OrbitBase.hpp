//
// Created by Sebastian on 12/8/2020.
//

#ifndef ORBITBASE_HPP_
#define ORBITBASE_HPP_

#include "FoldableBase.hpp"

class OrbitBase : public FoldableBase {
 public:
  void Fold(Eigen::Vector4f& p) const final  {}

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) const final {}

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const final {}

  void GLSL(GLSLFractalCode& buf) const override = 0;
  void UpdateUniforms(unsigned int ProgramID) const override = 0
};


#endif //ORBITBASE_HPP_
