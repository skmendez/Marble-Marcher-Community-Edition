//
// Created by Sebastian on 12/2/2020.
//

#ifndef FOLDSCALETRANSLATE_HPP_
#define FOLDSCALETRANSLATE_HPP_

#include <utility>

#include "FoldableBase.hpp"

class FoldScaleTranslate : public FoldableBase {
 public:
  FoldScaleTranslate(
      std::shared_ptr<GLSLVariable<float>> frac_scale,
      std::shared_ptr<GLSLVariable<Eigen::Vector3f>> frac_shift) :
      frac_scale_(std::move(frac_scale)), frac_shift_(std::move(frac_shift)
      ) {}

  void Fold(Eigen::Vector4f& p) const override {
    p *= frac_scale_->GetVar();
    p.segment<3>(0) += frac_shift_->GetVar();
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) const override {
    Fold(p);
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const override {
    n.segment<3>(0) -= frac_shift_->GetVar();
    n /= frac_scale_->GetVar();
  }

  void GLSL(GLSLFractalCode& buf) const override {
    buf << "p *= " << frac_scale_->GetGLSLVariable() << ";\n";
    buf << "p.xyz += " << frac_shift_->GetGLSLVariable() << ";\n";
  }

 private:
  std::shared_ptr<GLSLVariable<float>> frac_scale_;
  std::shared_ptr<GLSLVariable<Eigen::Vector3f>> frac_shift_;
};


#endif //FOLDSCALETRANSLATE_HPP_
