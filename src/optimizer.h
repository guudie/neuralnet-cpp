#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#ifndef NEURALNET_H
#include "neuralnet.h"
#endif

class MSGD {
public:
    MSGD();
    ~MSGD();

    template<typename activation, typename out_activation, typename loss>
    static void fit(neuralnet<activation, out_activation, loss>& net, double rate, int epoch, int batch_size, double gamma = 0.9) {
        std::vector<Eigen::MatrixXd*> wv;
        std::vector<Eigen::VectorXd*> bv;
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
};

#endif