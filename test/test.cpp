#include "neuralnet.h"
#include "act_func.h"
#include <iostream>

int main() {
    neuralnet<linear> net;
    Eigen::VectorXd sampleInput(3);
    sampleInput << 1, 2, 3;

    net.addLayer(3);
    net.addOutput(1);

    net.initWeights();
    net.feedforward(sampleInput);

    std::cout << net.getOutput();
}