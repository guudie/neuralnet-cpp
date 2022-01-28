#include "neuralnet.h"
#include <iostream>
using namespace std;



int main() {
    Eigen::MatrixXd* lastW = new Eigen::MatrixXd(3, 2);
    (*lastW).array() = 1;
    Eigen::VectorXd next(2);
    next.array() = 200;
    Eigen::VectorXd o = (*lastW) * next;
    std::cout << o;
}