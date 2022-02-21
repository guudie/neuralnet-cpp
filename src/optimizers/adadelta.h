#ifndef ADADELTA_H
#define ADADELTA_H

#define EPS 1e-8

#include "../optimizers.h"

// adadelta
class AdaDelta : public optimizer {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

    std::unordered_map<mat*, mat> E_W;
    std::unordered_map<vec*, vec> E_b;

    const double gamma;

public:
    AdaDelta(const double& r, const double& g = 0.9) : optimizer(r), gamma(g) {}
    ~AdaDelta() {}

    // update the weights
    void update(const mat& dW, mat& W) {
        mat& E = E_W[&W];
        if(E.cols() != W.cols() || E.rows() != W.rows()) {
            E.resizeLike(W);
            E.setZero();
        }
        
        E.noalias() = E * gamma + dW.cwiseAbs2() * (1 - gamma);
        W.array()  -= (rate / (E.array() + EPS).sqrt()) * dW.array();
    }

    // update the biases
    void update(const vec& db, vec& b) {
        vec& E = E_b[&b];
        if(E.size() != b.size()) {
            E.resizeLike(b);
            E.setZero();
        }
        
        E.noalias() = E * gamma + db.cwiseAbs2() * (1 - gamma);
        b.array()  -= (rate / (E.array() + EPS).sqrt()) * db.array();
    }

    // name
    std::string name() {
        return "AdaDelta";
    }
};

#endif