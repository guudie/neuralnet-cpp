#ifndef MSGD_H
#define MSGD_H

#include "../optimizers.h"

// momentum based stochastic gradient descent
class MSGD : public optimizer {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

    std::unordered_map<mat*, mat> V_W;
    std::unordered_map<vec*, vec> V_b;

    const double gamma;

public:
    MSGD(const double& r, const double& g) : optimizer(r), gamma(g) {}
    ~MSGD() {}

    // update the weights
    void update(const mat& dW, mat& W) {
        mat& V = V_W[&W];
        if(V.cols() != W.cols() || V.rows() != W.rows()) {
            V.resizeLike(W);
            V.setZero();
        }
        
        V.noalias()  = V * gamma + dW * rate;
        W.noalias() -= V;
    }

    // update the biases
    void update(const vec& db, vec& b) {
        vec& V = V_b[&b];
        if(V.size() != b.size()) {
            V.resizeLike(b);
            V.setZero();
        }
        
        V.noalias()  = V * gamma + db * rate;
        b.noalias() -= V;
    }

    // name
    std::string name() {
        return "MSGD";
    }
};

#endif