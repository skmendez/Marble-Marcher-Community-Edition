//
// Created by Sebastian on 12/8/2020.
//

#ifndef STATICFRACTALS_HPP_
#define STATICFRACTALS_HPP_

#include <utility>

#include "FractalInclude.hpp"

std::unique_ptr<ObjectBase> BlackRepeatingCubes() {

  auto modulus_size = std::make_shared<GLSLConstant<float>>(1.0);
  auto smol_box = std::make_unique<ObjectBox>(std::make_shared<GLSLConstant<Eigen::Vector3f>>(Eigen::Vector3f(0.02, 0.02, 0.02)));

  std::vector<std::unique_ptr<FoldableBase>> mod_folds{};
  mod_folds.emplace_back(std::make_unique<FoldModulo>(Axis::X, modulus_size));
  mod_folds.emplace_back(std::make_unique<FoldModulo>(Axis::Y, modulus_size));
  mod_folds.emplace_back(std::make_unique<FoldModulo>(Axis::Z, modulus_size));

  auto mod_series = std::make_unique<FoldSeries>(std::move(mod_folds));

  return std::make_unique<Fractal>(std::move(mod_series), std::move(smol_box));
}

std::unique_ptr<ObjectBase> BlackRepeatingCubesInSphere() {
  auto cubes = BlackRepeatingCubes();
  auto sphere = std::make_unique<ObjectSphere>(std::make_shared<GLSLConstant<float>>(6.0));

  return std::make_unique<ObjectIntersect>(std::move(cubes), std::move(sphere));
}

std::unique_ptr<ObjectBase> MengerSponge(std::shared_ptr<GLSLUniform<int>> depth, std::shared_ptr<GLSLUniform<Eigen::Vector3f>> frac_color) {
  auto scale = std::make_shared<GLSLConstant<float>>(3.f);
  auto translate = std::make_shared<GLSLConstant<Eigen::Vector3f>>(Eigen::Vector3f(-2.f, -2.f, 0.f));
  auto plane = std::make_shared<GLSLConstant<Eigen::Vector3f>>(Eigen::Vector3f(0.f, 0.f, -1.f));
  auto offset = std::make_shared<GLSLConstant<float>>(-1.f);

  std::vector<std::unique_ptr<FoldableBase>> inner_folds{};
  inner_folds.emplace_back(std::make_unique<FoldAbs>());
  inner_folds.emplace_back(std::make_unique<FoldMenger>());
  inner_folds.emplace_back(std::make_unique<FoldScaleTranslate>(scale, translate));
  inner_folds.emplace_back(std::make_unique<FoldPlane>(plane, offset));
  inner_folds.emplace_back(std::make_unique<OrbitMax>(frac_color));

  auto series = std::make_unique<FoldSeries>(std::move(inner_folds));
  auto loop = std::make_unique<FoldRepeat>(depth, std::move(series));

  std::vector<std::unique_ptr<FoldableBase>> outer_elements{};

  outer_elements.emplace_back(std::make_unique<OrbitInit>(
      std::make_shared<GLSLConstant<Eigen::Vector3f>>(Eigen::Vector3f(0.0, 0.0, 0.0))
  ));
  outer_elements.emplace_back(std::move(loop));

  auto series2 = std::make_unique<FoldSeries>(std::move(outer_elements));

  auto box = std::make_unique<ObjectBox>(std::make_shared<GLSLConstant<Eigen::Vector3f>>(Eigen::Vector3f(1.f, 1.f, 1.f)));

  auto final_scale = std::make_shared<GLSLConstant<float>>(.33f);
  auto final_translate = std::make_shared<GLSLConstant<Eigen::Vector3f>>(Eigen::Vector3f(0.f, 0.f, 0.f));

  std::vector<std::unique_ptr<FoldableBase>> final_folds{};
  final_folds.emplace_back(std::make_unique<FoldScaleTranslate>(final_scale, final_translate));
  final_folds.emplace_back(std::move(series2));

  auto final_series = std::make_unique<FoldSeries>(std::move(final_folds));

  return std::make_unique<Fractal>(std::move(final_series), std::move(box));
}

std::unique_ptr<ObjectBase> MengerSphere(std::shared_ptr<GLSLUniform<int>> depth,  std::shared_ptr<GLSLUniform<Eigen::Vector3f>> frac_color) {
  auto sponge = MengerSponge(std::move(depth), std::move(frac_color));
  auto sphere = std::make_unique<ObjectSphere>(std::make_shared<GLSLConstant<float>>(3.f));

  return std::make_unique<ObjectDifference>(std::move(sponge), std::move(sphere));
}

#endif //STATICFRACTALS_HPP_
