#include "neuron.h"
#include <vector>
#include <eigen3/Eigen/Core>



class neuralnet {
private:

    std::vector<Eigen::MatrixXd*> weights;
    std::vector<Eigen::VectorXd*> layers;
    std::vector<Eigen::VectorXd*> errors;
    std::vector<Eigen::VectorXd*> biases;


public:
    neuralnet();
    ~neuralnet() {
        cleanWeights();
        cleanLayers();
        cleanErrors();
        cleanBiases();
    }

    void cleanWeights() {
        for(auto it : weights)
            delete it;
    }

    void cleanLayers() {
        for(auto it : layers)
            delete it;
    }

    void cleanErrors() {
        for(auto it : errors)
            delete it;
    }

    void cleanBiases() {
        for(auto it : biases)
            delete it;
    }


};