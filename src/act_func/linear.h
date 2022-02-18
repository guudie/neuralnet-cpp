#ifndef LINEAR_H
#define LINEAR_H

#ifndef NEURALNET_H
#include <eigen3/Eigen/Core>
#endif

class Linear {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

public:
    Linear();
    virtual ~Linear();

    // linear activation function (linear) σ(z) = z
    static void f(mat& a, const mat& z) {
        a.noalias() = z;
    }

    // calculate [∂E / ∂z] = [∂E / ∂a] * σ'(z) and store it
    static void apply_diff(mat& dz, const mat& da, const mat& z, const mat& a) {
        // linear: [∂E / ∂z] = [∂E / ∂a] * σ'(z) = [∂E / ∂a] * 1
        dz.noalias() = da;
    }

    static string name() {
        return "Linear";
    }
};

#endif