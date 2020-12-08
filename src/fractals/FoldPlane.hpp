//
// Created by Sebastian on 12/8/2020.
//

#ifndef FOLDPLANE_HPP_
#define FOLDPLANE_HPP_

class FoldPlane : public FoldableBase {
 public:
  FoldPlane(std::shared_ptr<GLSLVariable<Eigen::Vector3f>> normal, std::shared_ptr<GLSLVariable<float>> offset) :
      normal_(std::move(normal)),
      offset_(std::move(offset)) {}

  void Fold(Eigen::Vector4f& p) const override {
    auto norm = normal_->GetVar();
    p.segment<3>(0) -= 2.f * std::min(0.f, p.segment<3>(0).dot(norm) - offset_->GetVar()) * norm;
  }

  void Fold(Eigen::Vector4f& p, FoldHistory& p_hist) const override {
    p_hist.push_back(p);
    Fold(p);
  }

  void Unfold(FoldHistory& p_hist, Eigen::Vector3f& n) const override {
    Eigen::Vector4f p = p_hist.back(); p_hist.pop_back();
    auto norm = normal_->GetVar();
    if (p.segment<3>(0).dot(norm) - offset_->GetVar() < 0.f) {
      n -= 2.f * (n.dot(norm) - offset_->GetVar()) * norm;
    }
  }

  void GLSL(GLSLFractalCode& buf) const override {
    buf << "planeFold(p, " << normal_->GetGLSLVariable() << ", " << offset_->GetGLSLVariable() << ");\n";
  }

  void UpdateUniforms(unsigned int ProgramID) const override {
    normal_->UpdateUniform(ProgramID);
    offset_->UpdateUniform(ProgramID);
  }

 private:
  std::shared_ptr<GLSLVariable<Eigen::Vector3f>> normal_;
  std::shared_ptr<GLSLVariable<float>> offset_;
};

#endif //FOLDPLANE_HPP_
