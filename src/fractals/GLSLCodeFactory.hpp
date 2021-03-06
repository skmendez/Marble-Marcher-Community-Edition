//
// Created by Sebastian on 12/7/2020.
//

#ifndef GLSLCODEFACTORY_HPP_
#define GLSLCODEFACTORY_HPP_

#include "ObjectBase.hpp"
#include "GLSLBase.hpp"

class GLSLCodeFactory {
 public:
  static std::string GenerateDistanceEstimator(const ObjectBase& obj) {
    GLSLFractalCode buf{};

    buf << "float de_fractal(vec4 p) {" << std::endl;
    buf.IncreaseIndent();
    buf << "float d = 1.0 / 0.0;" << std::endl;
    obj.GLSL(buf);
    buf << "return d;" << std::endl;
    buf.DecreaseIndent();
    buf << "}" << std::endl;
    return buf.get();
  }

  static std::string GenerateColor(const ObjectBase& obj) {
    GLSLFractalCode buf{};
    buf.SetPassType(true);

    buf << "vec4 col_fractal(vec4 p) {" << std::endl;
    buf.IncreaseIndent();
    buf << "float d = 1.0 / 0.0;" << std::endl;
    buf << "vec3 orbit = vec3(0.0);" << std::endl;
    obj.GLSL(buf);
    buf << "return vec4(orbit, d);" << std::endl;
    buf.DecreaseIndent();
    buf << "}" << std::endl;
    return buf.get();
  }
};


#endif //GLSLCODEFACTORY_HPP_
