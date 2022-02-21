#ifndef RELU_H
#define RELU_H

#include <eigen3/Eigen/Core>

class ReLU {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

public:
    ReLU();
    virtual ~ReLU();

    // relu activation function σ(z) = z if z > 0; 0 otherwise
    static void f(mat& a, const mat& z) {
        a.noalias() = z.cwiseMax(0);
    }

    // calculate [∂E / ∂z] = [∂E / ∂a] * σ'(z) and store it
    static void apply_diff(mat& dz, const mat& da, const mat& z, const mat& a) {
        // relu: [∂E / ∂z] = [∂E / ∂a] * σ'(z)
        // σ'(z) = (bool) z > 0
        dz.noalias() = (z.array() > 0).select(da, 0);
    }

    static std::string name() {
        return "ReLU";
    }
};

#endif