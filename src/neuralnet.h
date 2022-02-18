#ifndef NEURALNET_H
#define NEURALNET_H

#include <eigen3/Eigen/Core>
#include <vector>

#include "layer.h"

class neuralnet {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;
    typedef std::vector<vec*> vecContainer;

    std::vector<layer*> network;

    vecContainer train_in;
    vecContainer train_out;

    const double rate;
    const int steps;
    const int batch_size;

public:
    neuralnet(const double& r = 0.01, const int& s = 1000, const int& bs = 32) : rate(r), steps(s), batch_size(bs) {}
    ~neuralnet() {
        for(auto it : network)
            delete it;
        for(auto it : train_in)
            delete it;
        for(auto it : train_out)
            delete it;
    }

    // return the layers
    const std::vector<layer*>& getNetwork() {
        return network;
    }

    // add a layer
    void addLayer(layer* L) {
        network.push_back(L);
    }

    // feedforward algorithm
    void feedforward(const mat& data_in) {
        network[0]->evaluate(data_in);
        for(int l = 1; l < network.size(); l++) {
            network[l]->evaluate(network[l-1]->getData());
        }
    }
};

#endif