#ifndef ACT_FUNC_H
#define ACT_FUNC_H

#ifndef NEURALNET_H
#include <eigen3/Eigen/Core>
#endif

class Linear {
public:
    Linear();
    ~Linear();

    static double f(const double& x) {
        return x;
    }

    static void f(Eigen::VectorXd* layer, const Eigen::VectorXd* term) {
        for(int i = 0; i < term->size(); i++) {
            layer->coeffRef(i) = f(term->coeff(i));
        }
    }

    static double diff(const double&) {
        return 1;
    }

    static std::string name() {
        return "Linear";
    }
};

class ReLU {
public:
    ReLU();
    ~ReLU();

    static double f(const double& x) {
        if(x < 0)
            return 0;
        return x;
    }

    static void f(Eigen::VectorXd* layer, const Eigen::VectorXd* term) {
        for(int i = 0; i < term->size(); i++) {
            layer->coeffRef(i) = f(term->coeff(i));
        }
    }

    static double diff(const double& x) {
        if(x < 0)
            return 0;
        return 1;
    }

    static std::string name() {
        return "ReLU";
    }
};

class Sigmoid {
public:
    Sigmoid();
    ~Sigmoid();

    static double f(const double& x) {
        return 1 / (1 + exp(-x));
    }

    static void f(Eigen::VectorXd* layer, const Eigen::VectorXd* term) {
        for(int i = 0; i < term->size(); i++) {
            layer->coeffRef(i) = f(term->coeff(i));
        }
    }

    static double diff(const double& x) {
        double tmp = f(x);
        return tmp - tmp*tmp;
    }

    static std::string name() {
        return "Sigmoid";
    }
};

class Fastmoid {
public:
    Fastmoid();
    ~Fastmoid();

    static double f(const double& x) {
        return (x / (1 + abs(x)) + 1) / 2;
    }

    static void f(Eigen::VectorXd* layer, const Eigen::VectorXd* term) {
        for(int i = 0; i < term->size(); i++) {
            layer->coeffRef(i) = f(term->coeff(i));
        }
    }

    static double diff(const double& x) {
        double tmp = 1 + abs(x);
        return 1/tmp/tmp/2;
    }

    static std::string name() {
        return "Fastmoid";
    }
};

class Tanh {
public:
    Tanh();
    ~Tanh();

    static double f(const double& x) {
        return tanh(x);
    }

    static void f(Eigen::VectorXd* layer, const Eigen::VectorXd* term) {
        for(int i = 0; i < term->size(); i++) {
            layer->coeffRef(i) = f(term->coeff(i));
        }
    }

    static double diff(const double& x) {
        double tmp = cosh(x);
        return 1/tmp/tmp;
    }

    static std::string name() {
        return "Tanh";
    }
};

class Softmax {
public:
    Softmax();
    ~Softmax();

    
};

#endif