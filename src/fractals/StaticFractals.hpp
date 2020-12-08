//
// Created by Sebastian on 12/8/2020.
//

#ifndef STATICFRACTALS_HPP_
#define STATICFRACTALS_HPP_

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

#endif //STATICFRACTALS_HPP_
