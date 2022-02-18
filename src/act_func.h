#ifndef ACT_FUNC_H
#define ACT_FUNC_H

#ifndef NEURALNET_H
#include <eigen3/Eigen/Core>
#endif

class activation {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

public:
    activation();
    virtual ~activation();

    // default activation function (linear) σ(z) = z
    static void f(mat& a, const mat& z) {
        a.noalias() = z;
    }

    // default activation derivative (linear) σ'(z) = 1
    static void diff(mat& daz, const mat& z) {
        daz.array() = 1;
    }
};

#endif