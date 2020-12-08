//
// Created by Sebastian on 12/1/2020.
//

#ifndef FOLDABS_HPP_
#define FOLDABS_HPP_

#include <algorithm>
#include "FoldableBase.hpp"

class FoldAbs : public FoldableBase {
 public:
  FoldAbs() {}

  void Fold(Eigen::Vector4f& p) const override {
    p.segment<3>(0) = p.segment<3>(0).cwiseAbs();
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) const override {
    p_hist.push_back(p);
    Fold(p);
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const override {
    Eigen::Vector4f p = p_hist.back(); p_hist.pop_back();

    if (p[0] < 0.0f) {
      n[0] = -n[0];
    }
    if (p[1] < 0.0f) {
      n[1] = -n[1];
    }
    if (p[2] < 0.0f) {
      n[2] = -n[2];
    }
  }

  void GLSL(GLSLFractalCode& buf) const override {
    buf << "p.xyz = abs(p.xyz);\n";
  }

  void UpdateUniforms(unsigned int ProgramID) const override {}
};


#endif //FOLDABS_HPP_
