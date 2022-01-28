#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <eigen3/Eigen/Core>
#include <iostream>


template<typename activation>
class neuralnet {
private:

    std::vector<Eigen::MatrixXd*> weights;          // matrices of weights between layers
    std::vector<Eigen::MatrixXd*> gradient;

    std::vector<Eigen::VectorXd*> layers;           // includes input layer and all hidden layers
    std::vector<Eigen::VectorXd*> errors;           // dE/dn where E is cost, n is a neuron

    std::vector<Eigen::VectorXd*> biases;
    std::vector<Eigen::VectorXd*> biasErrors;
    std::vector<Eigen::VectorXd*> biasGradient;

    std::vector<Eigen::VectorXd*> train_in;
    std::vector<Eigen::VectorXd*> train_out;
    
    Eigen::VectorXd* output;

    bool validOut;


public:
    neuralnet() : validOut(false), output(NULL) {};
    ~neuralnet() {
        cleanWeights();
        cleanGradient();
        cleanLayers();
        //cleanOutput();    dont need this
        cleanErrors();
        cleanBiases();
        cleanBiasErrors();
        cleanBiasGradient();
        cleanTrain_in();
        cleanTrain_out();
        cleanOutput();
    }

    //
    // garbage collectors
    //
    void cleanWeights() {
        for(auto it : weights)
            if(it)
                delete it;
    }
    //
    void cleanGradient() {
        for(auto it : gradient)
            if(it)
                delete it;
    }
    //
    void cleanLayers() {
        for(auto it : layers)
            if(it)
                delete it;
    }
    //
    void cleanErrors() {
        for(auto it : errors)
            if(it)
                delete it;
    }
    //
    void cleanBiases() {
        for(auto it : biases)
            if(it)
                delete it;
    }
    //
    void cleanBiasErrors() {
        for(auto it : biasErrors)
            if(it)
                delete it;
    }
    //
    void cleanBiasGradient() {
        for(auto it : biasGradient)
            if(it)
                delete it;
    }
    //
    void cleanTrain_in() {
        for(auto it : train_in)
            if(it)
                delete it;
    }
    //
    void cleanTrain_out() {
        for(auto it : train_out)
            if(it)
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

    // X: all training input data
    // y: all training output
    void attach(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& y) {
        int n = X.size();
        int m = y.size();

        for(int i = 0; i < n; i++)
            train_in.push_back(new Eigen::VectorXd(X[i]));
        for(int i = 0; i < m; i++)
            train_out.push_back(new Eigen::VectorXd(y[i]));
    }

    void initWeights() {
        int n = layers.size();
        if(!n)
            return;
        
        errors.push_back(NULL);
        biases.push_back(NULL);
        biasErrors.push_back(NULL);
        biasGradient.push_back(NULL);
        // form weight matrices and biases between hidden layers
        for(int i = 0; i < n-1; i++) {
            // weights
            Eigen::MatrixXd* newWeight = new Eigen::MatrixXd(layers[i+1]->size(), layers[i]->size());
            newWeight->setZero();
            weights.push_back(newWeight);

            // gradient
            Eigen::MatrixXd* newGradient = new Eigen::MatrixXd(layers[i+1]->size(), layers[i]->size());
            newGradient->setZero();
            gradient.push_back(newGradient);

            // biases
            Eigen::VectorXd* newBias = new Eigen::VectorXd(layers[i+1]->size());
            newBias->setZero();
            biases.push_back(newBias);

            // bias errors
            Eigen::VectorXd* newBiasErr = new Eigen::VectorXd(layers[i+1]->size());
            newBiasErr->setZero();
            biasErrors.push_back(newBiasErr);

            // bias gradient
            Eigen::VectorXd* newBiasGrad = new Eigen::VectorXd(layers[i+1]->size());
            newBiasGrad->setZero();
            biasGradient.push_back(newBiasGrad);

            // errors
            Eigen::VectorXd* newErr = new Eigen::VectorXd(layers[i+1]->size());
            newErr->setZero();
            errors.push_back(newErr);
        }
    }

    void feedforward(const Eigen::VectorXd& input) {
        if(!weights.size() || input.size() != layers[0]->size())
            return;
        (*layers[0]) = input;

        int n = weights.size();
        for(int i = 0; i < n; i++) {
            (*layers[i+1]) = (*weights[i]) * (*layers[i]) + (*biases[i+1]);
            for(int j = 0; j < layers[i+1]->size(); j++)
                layers[i+1]->coeffRef(j) = activation::f(layers[i+1]->coeff(j));
        }
    }

    void clearGradient() {
        for(auto it : gradient)
            it->Zero();
        for(auto it : biasGradient)
            it->Zero();
    }

    void backprop(Eigen::VectorXd* const& y_i) {
        int L = layers.size() - 1;

        // calculate ∂E/∂ȳ_i
        (*errors[L]) = ((*output) - (*y_i)).array() * 2;

        // for each u_jk that affects a_j
        for(int j = 0; j < errors[L]->size(); j++) {
            double tmp = errors[L]->coeff(j) * activation::diff(layers[L]->coeff(j));
            biasGradient[L]->coeffRef(j) += tmp;
            for(int k = 0; k < layers[L-1]->size(); k++) {
                gradient[L-1]->coeffRef(j, k) += tmp * layers[L-1]->coeff(k);
            }
        }

        // calculate ∂E/∂a_i
        for(int l = L-1; l > 0; l--) {
            for(int j = 0; j < errors[l]->size(); j++) {
                errors[l]->coeffRef(j) = 0;
                for(int i = 0; i < errors[l+1]->size(); i++) {
                    double tmp = errors[l+1]->coeff(i) * activation::diff(layers[l+1]->coeff(i));
                    errors[l]->coeffRef(j) += tmp * weights[l]->coeff(i, j);
                }
                // for each u_jk that affects a_j
                double tmp0 = errors[l]->coeff(j) * activation::diff(layers[l]->coeff(j));
                biasGradient[l]->coeffRef(j) += tmp0;
                for(int k = 0; k < layers[l-1]->size(); k++) {
                    gradient[l-1]->coeffRef(j, k) += tmp0 * layers[l-1]->coeff(k);
                }
            }
        }
    }

    void fit(double rate, int epoch, int batch_size = 0) {
        for(int i = 0; i < epoch; i++) {
            clearGradient();
            for(int j = 0; j < batch_size; j++) {
                feedforward(*train_in[j]);
                backprop(train_out[j]);
            }

            for(int j = 0; j < gradient.size(); j++) {
                (*gradient[j]) = gradient[j]->array() / batch_size * rate;
                (*weights[j]) -= *gradient[j];
            }
            for(int j = 1; j < biasGradient.size(); j++) {
                (*biasGradient[j]) = biasGradient[j]->array() / batch_size * rate;
                (*biases[j]) -= *biasGradient[j];
            }
        }
    }
};

#endif