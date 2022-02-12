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
        std::vector<Eigen::MatrixXd*> v;
        for(auto a : net.gradient) {
            Eigen::MatrixXd* tmp = new Eigen::MatrixXd(a->rows(), a->cols());
            tmp->setZero();

            v.push_back(tmp);
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
                v[j]->noalias() = (*v[j]) * gamma + (*net.gradient[j]) * rb;
                net.weights[j]->noalias() -= *v[j];
            }
            for(int j = 1; j < net.biasGradient.size(); j++) {
                net.biases[j]->noalias() -= (*net.biasGradient[j]) * rb;
            }
        }

        for(auto a : v) {
            delete a;
        }
    }
};

#endif