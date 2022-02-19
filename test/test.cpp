#include "log.h"

#include "inc.h"
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    neuralnet net(0.0001, 10000, 2);
    net.addLayer(new dense<Linear>(3, 1));

    Eigen::MatrixXd in(3, 2);
    in <<   1, 4,
            2, 5,
            3, 6;
    net.randInit();
    net.feedforward(in);
    cout << net.getNetwork()[0]->getData();
}