//
// Created by Sebastian on 12/2/2020.
//

#ifndef FOLDREPEAT_HPP_
#define FOLDREPEAT_HPP_

#include "FoldableBase.hpp"
class FoldRepeat : public FoldableBase {
 public:
  FoldRepeat(uint16_t iterations, std::unique_ptr<FoldableBase> inner_fold) :
  iterations_(iterations), inner_fold_(std::move(inner_fold)) {}

  void Fold(Eigen::Vector4f& p) override {
    for (int i = 0; i < iterations_; i++) {
      inner_fold_->Fold(p);
    }
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) override {
    for (int i = 0; i < iterations_; i++) {
      inner_fold_->Fold(p, p_hist);
    }
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) override {
    for (int i = 0; i < iterations_; i++) {
      inner_fold_->Unfold(p_hist, n);
    }
  }

  void GLSL(IndentableOStreamBuf& buf) override {
    static int depth = 0;
    std::string var_name = "iter_";
    var_name.push_back('i' + depth);
    depth++;
    buf << "for (int " << var_name << " = 0; " << var_name << " < " << iterations_ << "; " << var_name << "++) {" << std::endl;
    buf.IncreaseIndent();
    inner_fold_->GLSL(buf);
    buf.DecreaseIndent();
    buf << "}" << std::endl;
  }


 private:
  const uint16_t iterations_;
  std::unique_ptr<FoldableBase> inner_fold_;
};


#endif //FOLDREPEAT_HPP_
