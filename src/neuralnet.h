#include <vector>
#include <eigen3/Eigen/Core>


template<typename activation>
class neuralnet {
private:

    std::vector<Eigen::MatrixXd*> weights;          // matrices of weights between layers
    std::vector<Eigen::VectorXd*> layers;           // includes input layer and all hidden layers
    std::vector<Eigen::VectorXd*> errors;           // dE/dn where E is cost, n is a neuron
    std::vector<Eigen::VectorXd*> biases;
    Eigen::VectorXd* output;
    bool validIn;


public:
    neuralnet() : validIn(false) {};
    ~neuralnet() {
        cleanWeights();
        cleanLayers();
        cleanOutput();
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

    
    void addLayer(int _size) {
        if(_size <= 0)
            return;
        Eigen::VectorXd* newLayer = new Eigen::VectorXd(_size);
        layers.push_back(newLayer);
    }

    void addInput(int _size) {
        addLayer(_size);
        if(layers.size())
            validIn = true;
    }

    void addOutput(int _size) {
        if(_size <= 0)
            return;
        cleanOutput();
        output = new Eigen::VectorXd(_size);
    }

    void initWeights() {
        int n = layers.size();
        if(!n)
            return;
        
        // form weight matrices between input and first hidden layer
        // and between hidden layers
        for(int i = 0; i < n-1; i++) {
            Eigen::MatrixXd* newWeight = new Eigen::MatrixXd(layers[i+1]->size(), layers[i]->size());
            newWeight->setZero();
            weights.push_back(newWeight);
        }

        Eigen::MatrixXd* lastW = new Eigen::MatrixXd(output->size(), layers[n-1]->size());
        lastW->setZero();
        weights.push_back(lastW);
    }

    void feedforward(const Eigen::VectorXd& input) {
        if(!validIn || !output)
            return;
        (*layers[0]) = input;

        int n = weights.size();
        for(int i = 0; i < n; i++) {
            (*layers[i+1]) = (*weights[i]) * (*layers[i]);
        }
    }
};