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

    template<typename A, typename B, typename C>
    static void fit(neuralnet<A, B, C>& net, double rate, int epoch, int batch_size, double gamma = 0.9) {
        matContainer wv;
        vecContainer bv;
        bv.push_back(NULL);
        for(auto a : net.gradient) {
            Eigen::MatrixXd* tmp_w = new Eigen::MatrixXd(a->rows(), a->cols());
            tmp_w->setZero();
            wv.push_back(tmp_w);

            Eigen::VectorXd* tmp_b = new Eigen::VectorXd(a->rows());
            tmp_b->setZero();
            bv.push_back(tmp_b);
        }

        int it = 0;
        int sz = net.train_in.size();
        double rb = rate / batch_size;
        for(int i = 0; i < epoch; i++) {
            net.clearGradient();
            for(int j = 0; j < batch_size; j++) {
                net.feedforward(*net.train_in[(it + j) % sz]);
                net.backprop(net.train_out[(it + j) % sz], batch_size);
            }
            it = (it + batch_size) % sz;

            for(int j = 0; j < net.gradient.size(); j++) {
                wv[j]->noalias() = (*wv[j]) * gamma + (*net.gradient[j]) * rb;
                net.weights[j]->noalias() -= *wv[j];
            }
            for(int j = 1; j < net.biasGradient.size(); j++) {
                bv[j]->noalias() = (*bv[j]) * gamma + (*net.biasGradient[j]) * rb;
                net.biases[j]->noalias() -= *bv[j];
            }
        }

        for(int i = 0; i < wv.size(); i++) {
            delete wv[i];
            delete bv[i+1];
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

    template<typename A, typename B, typename C>
    static void fit(neuralnet<A, B, C>& net, double rate, int epoch, int batch_size, double gamma = 0.9) {
        matContainer wG;
        vecContainer bG;
        bG.push_back(NULL);
        for(auto a : net.gradient) {
            Eigen::MatrixXd* tmp_w = new Eigen::MatrixXd(a->rows(), a->cols());
            tmp_w->setZero();
            wG.push_back(tmp_w);

            Eigen::VectorXd* tmp_b = new Eigen::VectorXd(a->rows());
            tmp_b->setZero();
            bG.push_back(tmp_b);
        }

        int it = 0;
        int sz = net.train_in.size();
        double rb = rate / batch_size;
        for(int i = 0; i < epoch; i++) {
            net.clearGradient();
            for(int j = 0; j < batch_size; j++) {
                net.feedforward(*net.train_in[(it + j) % sz]);
                net.backprop(net.train_out[(it + j) % sz], batch_size);
            }
            it = (it + batch_size) % sz;

            for(int l = 0; l < net.gradient.size(); l++) {
                for(int i = 0; i < net.gradient[l]->rows(); i++) {
                    for(int j = 0; j < net.gradient[l]->cols(); j++) {
                        wG[l]->coeffRef(i, j) += net.gradient[l]->coeff(i, j) * net.gradient[l]->coeff(i, j);
                        double num = rb * invSqrt(wG[l]->coeff(i, j) + EPS);
                        net.weights[l]->coeffRef(i, j) -= net.gradient[l]->coeff(i, j) * num;
                    }
                }

                for(int i = 0; i < net.biasGradient[l+1]->size(); i++) {
                    bG[l+1]->coeffRef(i) += net.biasGradient[l+1]->coeff(i) * net.biasGradient[l+1]->coeff(i);
                    double num = rb * invSqrt(bG[l+1]->coeff(i) + EPS);
                    net.biases[l+1]->coeffRef(i) -= net.biasGradient[l+1]->coeff(i) * num;
                }
            }
        }

        for(int i = 0; i < wG.size(); i++) {
            delete wG[i];
            delete bG[i+1];
        }
    }

    static std::string name(){
        return "AdaGrad";
    }
};

#endif