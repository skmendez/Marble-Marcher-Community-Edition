//
// Created by Sebastian on 12/8/2020.
//

#ifndef FOLDMODULO_HPP_
#define FOLDMODULO_HPP_

#include <utility>

#include "FoldableBase.hpp"

class FoldModulo : public FoldableBase {
 public:
  FoldModulo(const Axis modulo_axis, std::shared_ptr<GLSLVariable<float>> modulus) :
      modulo_axis_(modulo_axis), modulus_(std::move(modulus)) {}

  void Fold(Eigen::Vector4f& p) const override {
    float m = modulus_->GetVar();
    p(modulo_axis_) = std::abs(fmodulo(p(modulo_axis_) - m / 2.f, m) - m / 2.f);
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) const override {
    p_hist.push_back(p);
    Fold(p);
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const override {
    Eigen::Vector4f p = p_hist.back();
    p_hist.pop_back();
    float m = modulus_->GetVar();
    float a = fmodulo((p(modulo_axis_) - m / 2.f), m) - m / 2.f;
    if (a < 0.0) {
      n(modulo_axis_) = -n(modulo_axis_);
    }
    n(modulo_axis_) += p(modulo_axis_) - a;
  }

  void GLSL(GLSLFractalCode& buf) const override {
    std::string var;
    switch (modulo_axis_) {
      case Axis::X:
        var = "x";
        break;
      case Axis::Y:
        var = "y";
        break;
      case Axis::Z:
        var = "z";
        break;
    }
    std::string m = modulus_->GetGLSLVariable();
    buf << "p." << var << " = abs(mod(p." << var << " - " << m << "/2.0, " << m << ") - " << m << "/2.0);\n";
  }

  void UpdateUniforms(unsigned int ProgramID) const override {
    modulus_->UpdateUniform(ProgramID);
  }

 private:
  static inline float fmodulo(float a, float b) {
      const float result = std::fmod(a, b);
      return result >= 0.f ? result : result + b;
  }

  const Axis modulo_axis_;
  std::shared_ptr<GLSLVariable<float>> modulus_;
};


#endif //FOLDMODULO_HPP_
