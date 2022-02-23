#ifndef TANH_H
#define TANH_H

#include <eigen3/Eigen/Core>

class Tanh {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

public:
    Tanh();
    virtual ~Tanh();

    // tanh activation function σ(z) = tanh(z)
    static void f(mat& a, const mat& z) {
        a.array() = z.array().tanh();
    }

    // calculate [∂E / ∂z] = [∂E / ∂a] * σ'(z) and store it
    static void apply_diff(mat& dz, const mat& da, const mat& z, const mat& a) {
        // tanh: [∂E / ∂z] = [∂E / ∂a] * σ'(z) = [∂E / ∂a] * (1 - σ²(z))
        dz.array() = da.array() * (1 - a.array().square());
    }

    static std::string name() {
        return "Tanh";
    }
};

#endif