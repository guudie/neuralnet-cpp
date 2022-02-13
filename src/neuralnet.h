#ifndef NEURALNET_H
#define NEURALNET_H

#include "utils.h"
#include <eigen3/Eigen/Core>
#include <vector>


template<typename activation, typename out_activation, typename loss>
class neuralnet {
private:

    std::vector<Eigen::MatrixXd*> weights;          // matrices of weights between layers
    std::vector<Eigen::MatrixXd*> gradient;

    std::vector<Eigen::VectorXd*> terms;            // linear term: W * a + b
    std::vector<Eigen::VectorXd*> layers;           // includes input layer and all hidden layers
    std::vector<Eigen::VectorXd*> errors;           // dE/dn where E is cost, n is a neuron

    std::vector<Eigen::VectorXd*> biases;
    std::vector<Eigen::VectorXd*> biasGradient;

    std::vector<Eigen::VectorXd*> train_in;
    std::vector<Eigen::VectorXd*> train_out;
    
    Eigen::VectorXd* output;

    bool validOut;


    friend class MSGD;

public:
    neuralnet() : validOut(false), output(NULL) {};
    ~neuralnet() {
        cleanAll();
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
    void cleanTerms() {
        for(auto it : terms)
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
        if(output) {
            delete output;
            validOut = false;
            output = NULL;
        }
    }
    
    void cleanAll() {
        cleanWeights();
        cleanGradient();
        cleanTerms();
        cleanLayers();
        // cleanOutput();    dont need this
        cleanErrors();
        cleanBiases();
        cleanBiasGradient();
        cleanTrain_in();
        cleanTrain_out();
        validOut = false;
    }

    const Eigen::VectorXd& getOutput() {
        return *output;
    }

    const std::vector<Eigen::MatrixXd*>& getWeights() {
        return weights;
    }

    const std::vector<Eigen::VectorXd*>& getTrainX() {
        return train_in;
    }

    const std::vector<Eigen::VectorXd*>& getLayers() {
        return layers;
    }

    const std::vector<Eigen::VectorXd*>& getBiases() {
        return biases;
    }
    
    void addLayer(int _size) {
        if(_size <= 0 || validOut)
            return;
        
        Eigen::VectorXd* newLayer = new Eigen::VectorXd(_size);
        layers.push_back(newLayer);

        Eigen::VectorXd* newTerm = new Eigen::VectorXd(_size);
        terms.push_back(newTerm);
    }

    void addOutput(int _size) {
        if(_size <= 0)
            return;
        if(validOut) {
            layers.pop_back();

            delete terms[terms.size()-1];
            terms.pop_back();
        }
        cleanOutput();

        output = new Eigen::VectorXd(_size);
        layers.push_back(output);

        Eigen::VectorXd* outTerm = new Eigen::VectorXd(_size);
        terms.push_back(outTerm);

        validOut = true;
    }
    
    void clearGradient() {
        for(auto it : gradient)
            if(it)
                it->setZero();
        for(auto it : biasGradient)
            if(it)
                it->setZero();
    }

    void randomize() {
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 eng(seed);
        std::uniform_real_distribution<double> urd(-.5, .5);
        // std::cout << weights[0]->cols() << "; " << weights[0]->rows() << "\n";
        for(auto it : weights) {
            for(int i = 0; i < it->rows(); i++)
                for(int j = 0; j < it->cols(); j++) {
                    it->coeffRef(i, j) = urd(eng);
                }
        }
        for(auto it : biases) {
            if(it)
                for(int i = 0; i < it->rows(); i++)
                    for(int j = 0; j < it->cols(); j++) {
                        it->coeffRef(i, j) = urd(eng);
                    }
        }
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

    void init() {
        int n = layers.size();
        if(!n)
            return;
        
        errors.push_back(NULL);
        biases.push_back(NULL);
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

            // bias gradient
            Eigen::VectorXd* newBiasGrad = new Eigen::VectorXd(layers[i+1]->size());
            newBiasGrad->setZero();
            biasGradient.push_back(newBiasGrad);

            // errors
            Eigen::VectorXd* newErr = new Eigen::VectorXd(layers[i+1]->size());
            newErr->setZero();
            errors.push_back(newErr);
        }
        randomize();
    }

    // feed forward algorithm
    // the output layer has to be calculated separately due to some limitations with softmax
    void feedforward(const Eigen::VectorXd& input) {
        if(!weights.size() || input.size() != layers[0]->size())
            return;
        
        // input layer
        (*layers[0]) = (*terms[0]) = input;

        // calculate and activate the hidden layers
        int n = weights.size();
        for(int i = 0; i < n-1; i++) {
            // calculate the terms: t[l] = W[l-1] * a[l-1] + b[l]
            terms[i+1]->noalias() = ((*weights[i]) * (*layers[i])).eval() + (*biases[i+1]);
            // activate those terms: a[l] = act_func(t[l])
            for(int j = 0; j < terms[i+1]->size(); j++)
                layers[i+1]->coeffRef(j) = activation::f(terms[i+1]->coeff(j));
        }

        // calculate and activate the output layer
        terms[n]->noalias() = ((*weights[n-1]) * (*layers[n-1])).eval() + (*biases[n]);
        out_activation::f(layers[n], terms[n]);
    }

    // backprop algorithm
    // NOTE:
    // ** output layer is calculated separately
    // ** errors[l] actually contains ∂E/∂a_i, but for optimization, at each step, all elements are multiplied by the act::diff of their respective terms[l]
    void backprop(Eigen::VectorXd* const& y_i, const int& batch_size) {
        int L = layers.size() - 1;

        // calculate ∂E/∂ȳ_i
        *errors[L] = loss::diff(*output, *y_i, batch_size);

        // for each u_jk that affects a_j
        for(int j = 0; j < errors[L]->size(); j++) {
            double tmp = errors[L]->coeff(j) * out_activation::diff(terms[L]->coeff(j));
            errors[L]->coeffRef(j) = tmp;                       // optimized, not actually ∂E/∂ȳ_i
            biasGradient[L]->coeffRef(j) += tmp;
            for(int k = 0; k < layers[L-1]->size(); k++) {
                gradient[L-1]->coeffRef(j, k) += tmp * layers[L-1]->coeff(k);
            }
        }

        for(int l = L-1; l > 0; l--) {
            for(int j = 0; j < errors[l]->size(); j++) {
                // calculate ∂E/∂a_i
                errors[l]->coeffRef(j) = 0;
                for(int i = 0; i < errors[l+1]->size(); i++) {
                    // double tmp = errors[l+1]->coeff(i) * activation::diff(terms[l+1]->coeff(i));
                    double tmp = errors[l+1]->coeff(i);         // optimized
                    errors[l]->coeffRef(j) += tmp * weights[l]->coeff(i, j);
                }
                
                // for each u_jk that affects a_j
                double tmp0 = errors[l]->coeff(j) * activation::diff(terms[l]->coeff(j));
                errors[l]->coeffRef(j) = tmp0;                  // optimized, not actually ∂E/∂a_i
                biasGradient[l]->coeffRef(j) += tmp0;
                for(int k = 0; k < layers[l-1]->size(); k++) {
                    gradient[l-1]->coeffRef(j, k) += tmp0 * layers[l-1]->coeff(k);
                }
            }
        }
    }

    // default fitting algorithm, uses SGD
    // if batch_size == -1, uses GD
    void fit(double rate, int epoch, int batch_size = -1) {
        if(layers.size() < 2)
            return;
        if(batch_size == -1)
            batch_size = train_in.size();

        int it = 0;
        int sz = train_in.size();
        double rb = rate / batch_size;
        for(int i = 0; i < epoch; i++) {
            clearGradient();
            for(int j = 0; j < batch_size; j++) {
                feedforward(*train_in[(it + j) % sz]);
                backprop(train_out[(it + j) % sz], batch_size);
            }
            it = (it + batch_size) % sz;

            for(int j = 0; j < gradient.size(); j++) {
                weights[j]->noalias() -= (*gradient[j]) * rb;
            }
            for(int j = 1; j < biasGradient.size(); j++) {
                biases[j]->noalias() -= (*biasGradient[j]) * rb;
            }
        }
    }

    // fit with optimizer, very experimental
    template<typename optimizer>
    void fit_with_optimizer(double rate, int epoch, int batch_size = -1) {
        if(layers.size() < 2)
            return;
        if(batch_size == -1)
            batch_size = train_in.size();
        
        optimizer::fit(*this, rate, epoch, batch_size);
    }

    template<typename regularizer>
    void fit_with_regularization(double rate, int epoch, int batch_size = -1, double lambda = -1) {
        if(layers.size() < 2)
            return;
        if(batch_size == -1)
            batch_size = train_in.size();
        if(lambda == -1)
            lambda = rate;
        
        int it = 0;
        int sz = train_in.size();
        double rb = rate / batch_size;
        double lb = lambda / batch_size;
        for(int i = 0; i < epoch; i++) {
            clearGradient();
            for(int j = 0; j < batch_size; j++) {
                feedforward(*train_in[(it + j) % sz]);
                backprop(train_out[(it + j) % sz], batch_size);
            }
            it = (it + batch_size) % sz;

            for(int j = 0; j < gradient.size(); j++) {
                for(int k = 0; k < weights[j]->rows(); k++)
                    for(int l = 0; l < weights[j]->cols(); l++)
                        weights[j]->coeffRef(k, l) -= gradient[j]->coeff(k, l) * rb + regularizer::diff(weights[j]->coeff(k, l)) * lb;
            }
            for(int j = 1; j < biasGradient.size(); j++) {
                biases[j]->noalias() -= (*biasGradient[j]) * rb;
            }
        }
    }
};

#endif