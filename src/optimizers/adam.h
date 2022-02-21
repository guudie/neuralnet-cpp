#ifndef ADAM_H
#define ADAM_H

#define EPS 1e-8

#include "../optimizers.h"

// adam
class Adam : public optimizer {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

    // weights equivalent of m and v
    std::unordered_map<mat*, mat> m_W;
    std::unordered_map<mat*, mat> v_W;
    // biases equivalent of m and v
    std::unordered_map<vec*, vec> m_b;
    std::unordered_map<vec*, vec> v_b;

    // stores power of beta1, beta2 for weights of layer mat*
    std::unordered_map<mat*, double> beta1t_W;
    std::unordered_map<mat*, double> beta2t_W;
    // stores power of beta1, beta2 for biases of layer vec*
    std::unordered_map<vec*, double> beta1t_b;
    std::unordered_map<vec*, double> beta2t_b;

    const double beta1;
    const double beta2;

public:
    Adam(const double& r, const double& b1 = 0.9, const double& b2 = 0.999) : optimizer(r), beta1(b1), beta2(b2) {}
    ~Adam() {}

    // update the weights
    void update(const mat& dW, mat& W) {
        mat& m = m_W[&W];
        mat& v = v_W[&W];
        double& beta1t = beta1t_W[&W];
        double& beta2t = beta2t_W[&W];
        if(m.cols() != W.cols() || m.rows() != W.rows()) {
            m.resizeLike(W);
            v.resizeLike(W);
            m.setZero();
            v.setZero();
            beta1t = 1;
            beta2t = 1;
        }
        
        // compute 2 momentums
        m.noalias() = m * beta1 + dW * (1 - beta1);
        v.noalias() = v * beta2 + dW.cwiseAbs2() * (1 - beta2);
        // update beta_t
        beta1t *= beta1;
        beta2t *= beta2;
        // mhat and vhat
        mat mhat = m / (1 - beta1t);
        mat vhat = v / (1 - beta2t);
        // update W
        W.array() -= (rate / (vhat.array().sqrt() + EPS)) * mhat.array();
    }

    // update the biases
    void update(const vec& db, vec& b) {
        vec& m = m_b[&b];
        vec& v = v_b[&b];
        double& beta1t = beta1t_b[&b];
        double& beta2t = beta2t_b[&b];
        if(m.size() != b.size()) {
            m.resizeLike(b);
            v.resizeLike(b);
            m.setZero();
            v.setZero();
            beta1t = 1;
            beta2t = 1;
        }
        
        // compute 2 momentums
        m.noalias() = m * beta1 + db * (1 - beta1);
        v.noalias() = v * beta2 + db.cwiseAbs2() * (1 - beta2);
        // update beta_t
        beta1t *= beta1;
        beta2t *= beta2;
        // mhat and vhat
        vec mhat = m / (1 - beta1t);
        vec vhat = v / (1 - beta2t);
        // update W
        b.array() -= (rate / (vhat.array().sqrt() + EPS)) * mhat.array();
    }

    // name
    std::string name() {
        return "Adam";
    }
};

#endif