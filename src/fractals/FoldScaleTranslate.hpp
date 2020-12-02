//
// Created by Sebastian on 12/2/2020.
//

#ifndef FOLDSCALETRANSLATE_HPP_
#define FOLDSCALETRANSLATE_HPP_

#include "FoldableBase.hpp"

class FoldScaleTranslate : public FoldableBase {
 public:
  FoldScaleTranslate(const float frac_scale, const Eigen::Vector3f& frac_shift) : frac_scale_(frac_scale),
                                                                                  frac_shift_(frac_shift) {}

  void Fold(Eigen::Vector4f& p) override {
    p *= frac_scale_;
    p.segment<3>(0) += frac_shift_;
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) override {
    Fold(p);
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) override {
    n.segment<3>(0) -= frac_shift_;
    n /= frac_scale_;
  }

 private:
  const float frac_scale_;
  const Eigen::Vector3f frac_shift_;
};


#endif //FOLDSCALETRANSLATE_HPP_
