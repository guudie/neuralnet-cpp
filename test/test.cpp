#include "neuralnet.h"
#include "act_func.h"
#include <iostream>

int main() {
    neuralnet<linear> net;

    net.addLayer(4);
    net.addOutput(1);

    net.initWeights();

    
}