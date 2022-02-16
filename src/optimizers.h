#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#ifndef NEURALNET_H
#include "neuralnet.h"
#endif

#define EPS 1e-8

class MSGD {
private:
    MSGD();
    typedef std::vector<Eigen::MatrixXd*> matContainer;
    typedef std::vector<Eigen::VectorXd*> vecContainer;

    matContainer wvt;
    vecContainer bvt;
    double gamma;
public:
    MSGD(const matContainer& gradient, const vecContainer& biasGrad) : gamma(0.9) {
        bvt.push_back(NULL);
        for(auto a : gradient) {
            Eigen::MatrixXd* tmp_w = new Eigen::MatrixXd(a->rows(), a->cols());
            tmp_w->setZero();
            wvt.push_back(tmp_w);

            Eigen::VectorXd* tmp_b = new Eigen::VectorXd(a->rows());
            tmp_b->setZero();
            bvt.push_back(tmp_b);
        }
    }
    ~MSGD() {
        for(auto a : wvt)
            delete a;
        for(auto a : bvt)
            if(a) delete a;
    }

    void setGamma(double _gamma) {
        gamma = _gamma;
    }

    void operator()(matContainer& weights, matContainer& gradient, vecContainer& biases, vecContainer& biasGrad, const double& rb) {
        for(int j = 0; j < gradient.size(); j++) {
            wvt[j]->noalias() = (*wvt[j]) * gamma + (*gradient[j]) * rb;
            weights[j]->noalias() -= *wvt[j];
        }
        for(int j = 1; j < biasGrad.size(); j++) {
            bvt[j]->noalias() = (*bvt[j]) * gamma + (*biasGrad[j]) * rb;
            biases[j]->noalias() -= *bvt[j];
        }
    }

    static std::string name(){
        return "MSGD";
    }
};

class AdaGrad {
private:
    AdaGrad();
    typedef std::vector<Eigen::MatrixXd*> matContainer;
    typedef std::vector<Eigen::VectorXd*> vecContainer;

    matContainer wGt;
    vecContainer bGt;
public:
    AdaGrad(const matContainer& gradient, const vecContainer& biasGrad) {
        bGt.push_back(NULL);
        for(auto a : gradient) {
            Eigen::MatrixXd* tmp_w = new Eigen::MatrixXd(a->rows(), a->cols());
            tmp_w->setZero();
            wGt.push_back(tmp_w);

            Eigen::VectorXd* tmp_b = new Eigen::VectorXd(a->rows());
            tmp_b->setZero();
            bGt.push_back(tmp_b);
        }
    }
    ~AdaGrad() {
        for(auto a : wGt)
            delete a;
        for(auto a : bGt)
            if(a) delete a;
    }

    void operator()(matContainer& weights, matContainer& gradient, vecContainer& biases, vecContainer& biasGrad, const double& rb) {
        for(int l = 0; l < gradient.size(); l++) {
            for(int i = 0; i < gradient[l]->rows(); i++) {
                for(int j = 0; j < gradient[l]->cols(); j++) {
                    wGt[l]->coeffRef(i, j) += gradient[l]->coeff(i, j) * gradient[l]->coeff(i, j);
                    double num = rb * invSqrt(wGt[l]->coeff(i, j) + EPS);
                    weights[l]->coeffRef(i, j) -= gradient[l]->coeff(i, j) * num;
                }
            }

            for(int i = 0; i < biasGrad[l+1]->size(); i++) {
                bGt[l+1]->coeffRef(i) += biasGrad[l+1]->coeff(i) * biasGrad[l+1]->coeff(i);
                double num = rb * invSqrt(bGt[l+1]->coeff(i) + EPS);
                biases[l+1]->coeffRef(i) -= biasGrad[l+1]->coeff(i) * num;
            }
        }
    }

    static std::string name(){
        return "AdaGrad";
    }
};

class AdaDelta {
private:
    AdaDelta();
    typedef std::vector<Eigen::MatrixXd*> matContainer;
    typedef std::vector<Eigen::VectorXd*> vecContainer;

    matContainer wEt;
    vecContainer bEt;
    double gamma;
public:
    AdaDelta(const matContainer& gradient, const vecContainer& biasGrad) : gamma(0.9) {
        bEt.push_back(NULL);
        for(auto a : gradient) {
            Eigen::MatrixXd* tmp_w = new Eigen::MatrixXd(a->rows(), a->cols());
            tmp_w->setZero();
            wEt.push_back(tmp_w);

            Eigen::VectorXd* tmp_b = new Eigen::VectorXd(a->rows());
            tmp_b->setZero();
            bEt.push_back(tmp_b);
        }
    }
    ~AdaDelta() {
        for(auto a : wEt)
            delete a;
        for(auto a : bEt)
            if(a) delete a;
    }

    void setGamma(double _gamma) {
        gamma = _gamma;
    }

    void operator()(matContainer& weights, matContainer& gradient, vecContainer& biases, vecContainer& biasGrad, const double& rb) {
        for(int l = 0; l < gradient.size(); l++) {
            for(int i = 0; i < gradient[l]->rows(); i++) {
                for(int j = 0; j < gradient[l]->cols(); j++) {
                    wEt[l]->coeffRef(i, j) = wEt[l]->coeff(i, j) * gamma + gradient[l]->coeff(i, j) * gradient[l]->coeff(i, j) * (1 - gamma);
                    double num = rb * invSqrt(wEt[l]->coeff(i, j) + EPS);
                    weights[l]->coeffRef(i, j) -= gradient[l]->coeff(i, j) * num;
                }
            }

            for(int i = 0; i < biasGrad[l+1]->size(); i++) {
                bEt[l+1]->coeffRef(i) = bEt[l+1]->coeff(i) * gamma + biasGrad[l+1]->coeff(i) * biasGrad[l+1]->coeff(i) * (1 - gamma);
                double num = rb * invSqrt(bEt[l+1]->coeff(i) + EPS);
                biases[l+1]->coeffRef(i) -= biasGrad[l+1]->coeff(i) * num;
            }
        }
    }

    static std::string name() {
        return "AdaDelta";
    }
};

class Adam {
private:
    Adam();
    typedef std::vector<Eigen::MatrixXd*> matContainer;
    typedef std::vector<Eigen::VectorXd*> vecContainer;

    matContainer wmt;
    matContainer wvt;
    vecContainer bmt;
    vecContainer bvt;
    double beta1, beta2;
    double beta1t, beta2t;
public:
    Adam(const matContainer& gradient, const vecContainer& biasGrad) : beta1(0.9), beta2(0.999), beta1t(0.9), beta2t(0.999) {
        bmt.push_back(NULL);
        bvt.push_back(NULL);
        for(auto a : gradient) {
            Eigen::MatrixXd* tmp_wm = new Eigen::MatrixXd(a->rows(), a->cols());
            Eigen::MatrixXd* tmp_wv = new Eigen::MatrixXd(a->rows(), a->cols());
            tmp_wm->setZero();
            tmp_wv->setZero();
            wmt.push_back(tmp_wm);
            wvt.push_back(tmp_wv);

            Eigen::VectorXd* tmp_bm = new Eigen::VectorXd(a->rows());
            Eigen::VectorXd* tmp_bv = new Eigen::VectorXd(a->rows());
            tmp_bm->setZero();
            tmp_bv->setZero();
            bmt.push_back(tmp_bm);
            bvt.push_back(tmp_bv);
        }
    }
    ~Adam() {
        for(auto a : wmt)
            delete a;
        for(auto a : wvt)
            delete a;
        for(auto a : bmt)
            if(a) delete a;
        for(auto a : bvt)
            if(a) delete a;
    }

    void operator()(matContainer& weights, matContainer& gradient, vecContainer& biases, vecContainer& biasGrad, const double& rb) {
        for(int l = 0; l < gradient.size(); l++) {
            for(int i = 0; i < gradient[l]->rows(); i++) {
                for(int j = 0; j < gradient[l]->cols(); j++) {
                    wmt[l]->coeffRef(i, j) = wmt[l]->coeff(i, j) * beta1 + gradient[l]->coeff(i, j) * (1 - beta1);
                    wvt[l]->coeffRef(i, j) = wvt[l]->coeff(i, j) * beta2 + gradient[l]->coeff(i, j) * gradient[l]->coeff(i, j) * (1 - beta2);
                    double m_hat = wmt[l]->coeff(i, j) / (1 - beta1t);
                    double v_hat = wvt[l]->coeff(i, j) / (1 - beta2t);
                    double num = rb / (sqrt(v_hat) + EPS);
                    weights[l]->coeffRef(i, j) -= m_hat * num;
                }
            }

            for(int i = 0; i < biasGrad[l+1]->size(); i++) {
                bmt[l+1]->coeffRef(i) = bmt[l+1]->coeff(i) * beta1 + biasGrad[l+1]->coeff(i) * (1 - beta1);
                bvt[l+1]->coeffRef(i) = bvt[l+1]->coeff(i) * beta2 + biasGrad[l+1]->coeff(i) * biasGrad[l+1]->coeff(i) * (1 - beta2);
                double m_hat = bmt[l+1]->coeff(i) / (1 - beta1t);
                double v_hat = bvt[l+1]->coeff(i) / (1 - beta2t);
                double num = rb / (sqrt(v_hat) + EPS);
                biases[l+1]->coeffRef(i) -= m_hat * num;
            }
        }
        beta1t *= beta1;
        beta2t *= beta2;
    }

    static std::string name() {
        return "Adam";
    }
};

#endif