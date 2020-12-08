//
// Created by Sebastian on 12/1/2020.
//

#ifndef FOLDABLEBASE_HPP_
#define FOLDABLEBASE_HPP_
#include <Eigen/Dense>
#include "GLSLBase.hpp"
#include "GLSLVariable.hpp"

enum Axis {X = 0, Y = 1, Z = 2};

using FoldHistory = std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>;

class FoldableBase : public GLSLBase {

 public:
  FoldableBase() = default;
  virtual void Fold(Eigen::Vector4f& p) const = 0;
  virtual void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) const {
    Fold(p);
  }

  virtual void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const = 0;
  virtual void GLSL(GLSLFractalCode& buf) const = 0;
  void UpdateUniforms(unsigned int ProgramID) const override = 0;
};

#endif //FOLDABLEBASE_HPP_
