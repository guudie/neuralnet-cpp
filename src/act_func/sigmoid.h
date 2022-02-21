#ifndef SIGMOID_H
#define SIGMOID_H

#include <eigen3/Eigen/Core>

class Sigmoid {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

public:
    Sigmoid();
    virtual ~Sigmoid();

    // sigmoid activation function σ(z) = 1 / (1 + exp(-z))
    static void f(mat& a, const mat& z) {
        a.array() = ((-z).array().exp() + 1).cwiseInverse();
    }

    // calculate [∂E / ∂z] = [∂E / ∂a] * σ'(z) and store it
    static void apply_diff(mat& dz, const mat& da, const mat& z, const mat& a) {
        // sigmoid: [∂E / ∂z] = [∂E / ∂a] * σ'(z) = [∂E / ∂a] * (σ(z) - σ²(z))
        dz.noalias() = da.cwiseProduct(a - a.cwiseAbs2());
    }

    static std::string name() {
        return "Sigmoid";
    }
};

#endif