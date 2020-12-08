//
// Created by Sebastian on 12/1/2020.
//

#ifndef FRACTAL_HPP_
#define FRACTAL_HPP_

#include <vector>
#include "FoldableBase.hpp"
#include "ObjectBase.hpp"

class Fractal : public ObjectBase {
 public:
  Fractal(std::unique_ptr<FoldableBase> fold, std::unique_ptr<ObjectBase> base) :
  fold_(std::move(fold)), base_(std::move(base)) {}

  float DistanceEstimator(Eigen::Vector4f p) const override {
    fold_->Fold(p);
    return base_->DistanceEstimator(p);
  }

  Eigen::Vector3f NearestPoint(Eigen::Vector4f p) const override {
    static FoldHistory p_hist;
    p_hist.clear();
    fold_->Fold(p, p_hist);
    Eigen::Vector3f n = base_->NearestPoint(p);
    fold_->Unfold(p_hist, n);
    return n;
  }

  void GLSL(GLSLFractalCode& buf) const override {
    fold_->GLSL(buf);
    base_->GLSL(buf);
  }

 private:

  std::unique_ptr<FoldableBase> fold_;
  std::unique_ptr<ObjectBase> base_;
};


#endif //FRACTAL_HPP_
