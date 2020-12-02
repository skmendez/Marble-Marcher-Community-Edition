//
// Created by Sebastian on 12/2/2020.
//

#ifndef FOLDSERIES_HPP_
#define FOLDSERIES_HPP_

#include "FoldableBase.hpp"

class FoldSeries : public FoldableBase {
 public:
  FoldSeries(std::vector<std::unique_ptr<FoldableBase>>inner_folds) :
    inner_folds_(std::move(inner_folds)) {}

  void Fold(Eigen::Vector4f& p) override {
    for (auto it = inner_folds_.begin(); it != inner_folds_.end(); ++it) {
      it->get()->Fold(p);
    }
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) override {
    for (auto it = inner_folds_.begin(); it != inner_folds_.end(); ++it) {
      it->get()->Fold(p, p_hist);
    }
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) override {
    for (auto it = inner_folds_.rbegin(); it != inner_folds_.rend(); ++it) {
      it->get()->Unfold(p_hist, n);
    }
  }

  void GLSL(IndentableOStreamBuf& buf) override {
    for (auto it = inner_folds_.begin(); it != inner_folds_.end(); ++it) {
      it->get()->GLSL(buf);
    }
  }

 private:
  std::vector<std::unique_ptr<FoldableBase>> inner_folds_;
};


#endif //FOLDSERIES_HPP_
