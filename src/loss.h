#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <eigen3/Eigen/Core>

class sse {
public:
    static Eigen::VectorXd diff(const Eigen::VectorXd& y_bar, const Eigen::VectorXd& y) {
        return (y_bar - y) * 2;
    }
};

#endif