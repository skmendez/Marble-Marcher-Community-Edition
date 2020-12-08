//
// Created by Sebastian on 12/1/2020.
//

#ifndef FOLDMENGER_HPP_
#define FOLDMENGER_HPP_



#include <algorithm>
#include "FoldableBase.hpp"

class FoldMenger : public FoldableBase {
 public:
  FoldMenger() = default;

  void Fold(Eigen::Vector4f& p) const override {
    float a = std::min(p.x() - p.y(), 0.0f);
    p.x() -= a; p.y() += a;
    a = std::min(p.x() - p.z(), 0.0f);
    p.x() -= a; p.z() += a;
    a = std::min(p.y() - p.z(), 0.0f);
    p.y() -= a; p.z() += a;
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) const override {
    p_hist.push_back(p);
    Fold(p);
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const override {
    Eigen::Vector4f p = p_hist.back(); p_hist.pop_back();

    const float mx = std::max(p[0], p[1]);
    if (std::min(p[0], p[1]) < std::min(mx, p[2])) {
      std::swap(n[1], n[2]);
    }
    if (mx < p[2]) {
      std::swap(n[0], n[2]);
    }
    if (p[0] < p[1]) {
      std::swap(n[0], n[1]);
    }
  }

  void GLSL(GLSLFractalCode& buf) const override {
    buf << "mengerFold(p);\n";
  }

  void UpdateUniforms(unsigned int ProgramID) const override {}
};


#endif //FOLDMENGER_HPP_
