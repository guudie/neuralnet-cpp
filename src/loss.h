#ifndef LOSS_H
#define LOSS_H

#ifndef NEURALNET_H
#include <vector>
#include <eigen3/Eigen/Core>
#endif

class sse {
public:
    static Eigen::VectorXd diff(const Eigen::VectorXd& y_bar, const Eigen::VectorXd& y, const int& batch_size) {
        return (y_bar - y) * 2;
    }
};

class mse {
public:
    static Eigen::VectorXd diff(const Eigen::VectorXd& y_bar, const Eigen::VectorXd& y, const int& batch_size) {
        return (y_bar - y) * (2 / batch_size);
    }
};

#endif