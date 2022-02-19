#ifndef NEURALNET_H
#define NEURALNET_H

#include <eigen3/Eigen/Core>
#include <vector>

#include "layer.h"

template<typename loss>
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

    // random init
    void randInit() {
        for(auto l : network)
            l->randInit(batch_size);
    }

    // attach training data
    void attach(const vecContainer& X, const vecContainer& y) {
        train_in.clear();
        train_out.clear();
        for(auto it : X)
            train_in.push_back(new vec(*it));
        for(auto it : y)
            train_out.push_back(new vec(*it));
    }

    // feedforward algorithm
    void feedforward(const mat& data_in) {
        network[0]->evaluate(data_in);
        for(int l = 1; l < network.size(); l++) {
            network[l]->evaluate(network[l-1]->getData());
        }
    }

    // backprop algorithm
    void backprop(const mat& X, const mat& y) {
        int L = network.size() - 1;
        // compute ∂E / ∂ȳ
        loss::diff(network[L]->daRef(), network[L]->getData(), y);
        // compute gradients from layer L --> 1
        for(int l = L; l > 0; l--) {
            network[l]->backprop(network[l-1]->getData(), network[l-1]->daRef());
        }
        // compute gradients between layers 0 and input
        network[0]->backprop(X);
    }

    // fitting algorithm
    void fit() {
        if(network.size() < 1 || batch_size < 1)
            return;
        
        mat tmp_in(train_in[0]->size(), batch_size);
        mat tmp_out(train_out[0]->size(), batch_size);
        int sz = train_in.size();
        int it = 0;
        for(int _ = 0; _ < steps; _++) {
            // attach #batch_size training observations to tmp
            for(int i = 0; i < batch_size; i++) {
                tmp_in.col(i) = *train_in[(i+it)%sz];
                tmp_out.col(i) = *train_out[(i+it)%sz];
            }
            it = (it + batch_size) % sz;

            feedforward(tmp_in);
            backprop(tmp_in, tmp_out);

            // update weights accordingly
            for(auto l : network)
                l->updateParams(rate);
        }
    }
};

#endif