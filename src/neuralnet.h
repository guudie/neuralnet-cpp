#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <stdio.h>


template<typename activation>
class neuralnet {
private:

    std::vector<Eigen::MatrixXd*> weights;          // matrices of weights between layers
    std::vector<Eigen::VectorXd*> layers;           // includes input layer and all hidden layers
    std::vector<Eigen::VectorXd*> errors;           // dE/dn where E is cost, n is a neuron
    std::vector<Eigen::VectorXd*> biases;
    Eigen::VectorXd* output;

    bool validOut;


public:
    neuralnet() : validOut(false), output(NULL) {};
    ~neuralnet() {
        cleanWeights();
        cleanLayers();
        //cleanOutput();    dont need this
        cleanErrors();
        cleanBiases();
    }

    //
    // garbage collectors
    //
    void cleanWeights() {
        for(auto it : weights)
            delete it;
    }
    //
    void cleanLayers() {
        for(auto it : layers)
            delete it;
    }
    //
    void cleanErrors() {
        for(auto it : errors)
            delete it;
    }
    //
    void cleanBiases() {
        for(auto it : biases)
            delete it;
    }
    //
    void cleanOutput() {
        if(output)
            delete output;
    }

    const Eigen::VectorXd& getOutput() {
        return *output;
    }

    const std::vector<Eigen::MatrixXd*>& getWeights() {
        return weights;
    }
    
    void addLayer(int _size) {
        if(_size <= 0)
            return;
        Eigen::VectorXd* newLayer = new Eigen::VectorXd(_size);
        layers.push_back(newLayer);
    }

    void addOutput(int _size) {
        if(_size <= 0)
            return;
        cleanOutput();
        output = new Eigen::VectorXd(_size);
        layers.push_back(output);
        validOut = true;
    }

    void initWeights() {
        int n = layers.size();
        if(!n)
            return;
        
        // form weight matrices and biases between hidden layers
        for(int i = 0; i < n-1; i++) {
            Eigen::MatrixXd* newWeight = new Eigen::MatrixXd(layers[i+1]->size(), layers[i]->size());
            newWeight->setZero();
            // newWeight->array() = 1;
            weights.push_back(newWeight);

            Eigen::VectorXd* newBias = new Eigen::VectorXd(layers[i+1]->size());
            newBias->setZero();
            // newBias->array() = 2;
            biases.push_back(newBias);
        }
    }

    void feedforward(const Eigen::VectorXd& input) {
        if(!weights.size() || input.size() != layers[0]->size())
            return;
        (*layers[0]) = input;

        int n = weights.size();
        for(int i = 0; i < n; i++) {
            (*layers[i+1]) = (*weights[i]) * (*layers[i]) + (*biases[i]);
            for(int j = 0; j < layers[i+1]->size(); j++)
                layers[i+1]->coeffRef(j) = activation::f(layers[i+1]->coeff(j));
        }
    }
};

#endif