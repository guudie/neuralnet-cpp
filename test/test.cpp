#include "log.h"

#include "inc.h"
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    layer* tmp_in = new dense<activation>(3, 1);
    tmp_in->init();

    neuralnet net(0.0001, 10000, 32);
    net.addLayer(tmp_in);

    Eigen::MatrixXd in(3, 2);
    in <<   1, 4,
            2, 5,
            3, 6;
    
    net.feedforward(in);
    cout << net.getNetwork()[0]->getData();
}