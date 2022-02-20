#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "utils.h"
#include <eigen3/Eigen/Core>
#include <unordered_map>
#include <algorithm>

class optimizer {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

protected:
    const double rate;

public:
    optimizer(const double& r) : rate(r) {}
    virtual ~optimizer() {}

    // update the weights
    virtual void update(const mat& dW, mat& W) = 0;

    // update the biases
    virtual void update(const vec& db, vec& b) = 0;

    // name
    virtual std::string name() = 0;
};

#endif