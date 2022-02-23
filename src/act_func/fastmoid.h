#ifndef FASTMOID_H
#define FASTMOID_H

#include <eigen3/Eigen/Core>

class Fastmoid {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

public:
    Fastmoid();
    virtual ~Fastmoid();

    // fastmoid activation function σ(z) = 0.5 * (z / (|z| + 1) + 1)
    static void f(mat& a, const mat& z) {
        a.array() = 0.5 * (z.array() * (z.cwiseAbs().array() + 1).cwiseInverse() + 1);
    }

    // calculate [∂E / ∂z] = [∂E / ∂a] * σ'(z) and store it
    static void apply_diff(mat& dz, const mat& da, const mat& z, const mat& a) {
        // fastmoid: [∂E / ∂z] = [∂E / ∂a] * σ'(z) = [∂E / ∂a] * 0.5/(|z| + 1)²
        dz.array() = da.array() * (z.cwiseAbs().array() + 1).cwiseAbs2().cwiseInverse() / 2;
    }

    static std::string name() {
        return "Fastmoid";
    }
};

#endif