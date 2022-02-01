#ifndef NEURALNET_H
#define NEURALNET_H

#ifdef DEBUG
#include <iostream>
#endif

#include <vector>
#include <eigen3/Eigen/Core>
#include <random>


template<typename activation, typename loss>
class neuralnet {
private:

    std::vector<Eigen::MatrixXd*> weights;          // matrices of weights between layers
    std::vector<Eigen::MatrixXd*> gradient;

    std::vector<Eigen::VectorXd*> layers;           // includes input layer and all hidden layers
    std::vector<Eigen::VectorXd*> errors;           // dE/dn where E is cost, n is a neuron

    std::vector<Eigen::VectorXd*> biases;
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
        // cleanOutput();    dont need this
        cleanErrors();
        cleanBiases();
        cleanBiasGradient();
        cleanTrain_in();
        cleanTrain_out();
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
    }

    void addOutput(int _size) {
        if(_size <= 0)
            return;
        if(validOut)
            layers.pop_back();
        cleanOutput();
        output = new Eigen::VectorXd(_size);
        layers.push_back(output);
        validOut = true;
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

    void initWeights() {
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

    void feedforward(const Eigen::VectorXd& input) {
        // std::cout << "feed-------\n";
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
            if(it)
                it->setZero();
        for(auto it : biasGradient)
            if(it)
                it->setZero();
    }

    void backprop(Eigen::VectorXd* const& y_i, const int& batch_size) {
        int L = layers.size() - 1;

        // calculate ∂E/∂ȳ_i
        (*errors[L]) = loss::diff(*output, *y_i, batch_size);

        // for each u_jk that affects a_j
        for(int j = 0; j < errors[L]->size(); j++) {
            // multiply each error of the last layer with their respective activation
            double tmp = errors[L]->coeff(j) * activation::diff(layers[L]->coeff(j));       // actually "out_activation" but will be fixed later
            errors[L]->coeffRef(j) = tmp;
            biasGradient[L]->coeffRef(j) += tmp;
            // for(int k = 0; k < layers[L-1]->size(); k++) {
            //     gradient[L-1]->coeffRef(j, k) += tmp * layers[L-1]->coeff(k);
            // }
        }
        *gradient[L-1] += (*errors[L]) * layers[L-1]->transpose();

        // calculate ∂E/∂a_i
        for(int l = L-1; l > 0; l--) {
            *errors[l] = weights[l]->transpose() * (*errors[l+1]);
            for(int j = 0; j < errors[l]->size(); j++) {
                // errors[l]->coeffRef(j) = 0;
                // for(int i = 0; i < errors[l+1]->size(); i++) {
                //     double tmp = errors[l+1]->coeff(i) * activation::diff(layers[l+1]->coeff(i));
                //     errors[l]->coeffRef(j) += tmp * weights[l]->coeff(i, j);
                // }

                // for each u_jk that affects a_j
                // double tmp0 = errors[l]->coeff(j) * activation::diff(layers[l]->coeff(j));
                // biasGradient[l]->coeffRef(j) += tmp0;
                // for(int k = 0; k < layers[l-1]->size(); k++) {
                //     gradient[l-1]->coeffRef(j, k) += tmp0 * layers[l-1]->coeff(k);
                // }

                double tmp = errors[l]->coeff(j) * activation::diff(layers[l]->coeff(j));
                errors[l]->coeffRef(j) = tmp;
                biasGradient[l]->coeffRef(j) += tmp;
            }
            *gradient[l-1] += (*errors[l]) * layers[l-1]->transpose();
        }
    }

    void fit(double rate, int epoch, int batch_size = -1) {
        if(layers.size() < 2)
            return;
        if(batch_size == -1)
            batch_size = train_in.size();
        // std::cout << "fit---------\n";
        for(int i = 0; i < epoch; i++) {
            clearGradient();
            // std::cout << "gradients cleared---------\n";
            for(int j = 0; j < batch_size; j++) {
                feedforward(*train_in[j]);
                backprop(train_out[j], batch_size);
            }
            // std::cout << "gradient: " << *gradient[0] << "\n";

            for(int j = 0; j < gradient.size(); j++) {
                (*weights[j]) -= (*gradient[j]) * (rate / batch_size);
            }
            for(int j = 1; j < biasGradient.size(); j++) {
                (*biases[j]) -= (*biasGradient[j]) * (rate / batch_size);
            }
        }
        // std::cout << "fitting done ----------\n";
    }
};

#endif