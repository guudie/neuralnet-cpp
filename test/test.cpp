#include "log.h"

#include "inc.h"
#include <iostream>
#include <fstream>
using namespace std;

int main() {
    layer* tmp_in = new dense<activation>(3, 1);
    tmp_in->init();

    neuralnet net;
    net.addLayer(tmp_in);

    Eigen::VectorXd in(3);
    in << 1, 2, 3;
    
    net.feedforward(in);
    cout << net.getNetwork()[0]->getData();
}