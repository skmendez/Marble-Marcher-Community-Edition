//
// Created by Sebastian on 12/1/2020.
//

#ifndef FOLDABLEBASE_HPP_
#define FOLDABLEBASE_HPP_
#include <Eigen/Dense>
#include "GLSLBase.hpp"

using FoldHistory = std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>;

class FoldableBase : public GLSLBase {

 public:
  FoldableBase() = default;
  virtual void Fold(Eigen::Vector4f& p) = 0;
  virtual void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) {
    Fold(p);
  }

  virtual void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) = 0;

  virtual void GLSL(IndentableOStreamBuf& buf) {
    // TODO: remove when everything has GLSL
  }
 private:

};

#endif //FOLDABLEBASE_HPP_
