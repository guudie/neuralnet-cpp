#ifndef SGD_H
#define SGD_H

#include "../optimizers.h"

// standard stochastic gradient descent
class SGD : public optimizer {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

public:
    SGD(const double& r) : optimizer(r) {}
    ~SGD() {}

    // update the weights
    void update(const mat& dW, mat& W) {
        W.noalias() -= dW * rate;
    }

    // update the biases
    void update(const vec& db, vec& b) {
        b.noalias() -= db * rate;
    }

    // name
    std::string name() {
        return "SGD";
    }
};

#endif