//
// Created by Sebastian on 12/2/2020.
//

#ifndef FOLDREPEAT_HPP_
#define FOLDREPEAT_HPP_

#include <utility>

#include "FoldableBase.hpp"
class FoldRepeat : public FoldableBase {
 public:
  FoldRepeat(std::shared_ptr<GLSLVariable<int>> iterations, std::unique_ptr<FoldableBase> inner_fold) :
  iterations_(std::move(iterations)), inner_fold_(std::move(inner_fold)) {}

  void Fold(Eigen::Vector4f& p) const override {
    for (int i = 0; i < iterations_->GetVar(); i++) {
      inner_fold_->Fold(p);
    }
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) const override {
    for (int i = 0; i < iterations_->GetVar(); i++) {
      inner_fold_->Fold(p, p_hist);
    }
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const override {
    for (int i = 0; i < iterations_->GetVar(); i++) {
      inner_fold_->Unfold(p_hist, n);
    }
  }

  void GLSL(GLSLFractalCode& buf) const override {
    static int depth = 0;
    std::string var_name = "iter_";
    var_name.push_back('i' + depth);
    depth++;
    buf << "for (int " << var_name << " = 0; " << var_name << " < " << iterations_->GetGLSLVariable() << "; " << var_name << "++) {\n";
    buf.IncreaseIndent();
    inner_fold_->GLSL(buf);
    buf.DecreaseIndent();
    buf << "}\n";
  }


 private:
  std::shared_ptr<GLSLVariable<int>> iterations_;
  std::unique_ptr<FoldableBase> inner_fold_;
};


#endif //FOLDREPEAT_HPP_
