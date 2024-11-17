#ifndef PARAM_RELU_H
#define PARAM_RELU_H

#include <eigen3/Eigen/Core>

class ParamReLU {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

    static constexpr double c = 0.25;
    static constexpr double absCoeff = -c/2 + 0.5;
    static constexpr double regCoeff = -c/2 - 0.5;

public:
    ParamReLU();
    virtual ~ParamReLU();

    // parametric relu activation function σ(z) = z if z > 0; cx otherwise
    static void f(mat& a, const mat& z) {
        a.noalias() = absCoeff * z.cwiseAbs() - regCoeff * z;
    }

    // calculate [∂E / ∂z] = [∂E / ∂a] * σ'(z) and store it
    static void apply_diff(mat& dz, const mat& da, const mat& z, const mat& a) {
        // parametric relu: [∂E / ∂z] = [∂E / ∂a] * σ'(z)
        // σ'(z) = 1 if z > 0; c otherwise
        dz.noalias() = (z.array() > 0).select(da, c * da);
    }

    static std::string name() {
        return "ParamReLU";
    }
};

#endif