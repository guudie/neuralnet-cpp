#ifndef ADAGRAD_H
#define ADAGRAD_H

#define EPS 1e-8

#include "../optimizers.h"

// adagrad
class AdaGrad : public optimizer {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

    std::unordered_map<mat*, mat> G_W;
    std::unordered_map<vec*, vec> G_b;

public:
    AdaGrad(const double& r) : optimizer(r) {}
    ~AdaGrad() {}

    // update the weights
    void update(const mat& dW, mat& W) {
        mat& G = G_W[&W];
        if(G.cols() != W.cols() || G.rows() != W.rows()) {
            G.resizeLike(W);
            G.setZero();
        }
        
        G.noalias() += dW.cwiseAbs2();
        W.array() -= (rate / (G.array() + EPS).sqrt()) * dW.array();
        // for(int i = 0; i < G.rows(); i++)
        //     for(int j = 0; j < G.cols(); j++) {
        //         G.coeffRef(i, j) += dW.coeff(i, j) * dW.coeff(i, j);
        //         double num = rate * invSqrt(G.coeff(i, j) + EPS);
        //         W.coeffRef(i, j) -= num * dW.coeff(i, j);
        //     }
    }

    // update the biases
    void update(const vec& db, vec& b) {
        vec& G = G_b[&b];
        if(G.size() != b.size()) {
            G.resizeLike(b);
            G.setZero();
        }
        
        G.noalias() += db.cwiseAbs2();
        b.array() -= (rate / (G.array() + EPS).sqrt()) * db.array();
        // for(int i = 0; i < G.size(); i++) {
        //     G.coeffRef(i) += db.coeff(i) * db.coeff(i);
        //     double num = rate * invSqrt(G.coeff(i) + EPS);
        //     b.coeffRef(i) -= num * db.coeff(i);
        // }
    }

    // name
    std::string name() {
        return "AdaGrad";
    }
};

#endif