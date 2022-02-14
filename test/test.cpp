#include "log.h"

#include "inc.h"
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

void readTrainingData(string dir, std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& y, int& n, int& ins, int& outs) {
    fstream fin(dir, ios::in);
    fin >> n >> ins >> outs;
    for(int i = 0; i < n; i++) {
        double x_i;
        double y_i;
        Eigen::VectorXd tmp_x(ins);
        Eigen::VectorXd tmp_y(outs);

        for(int i = 0; i < ins; i++) {
            fin >> x_i;
            tmp_x.coeffRef(i) = x_i;
        }
        for(int i = 0; i < outs; i++) {
            fin >> y_i;
            tmp_y.coeffRef(i) = y_i;
        }

        X.push_back(tmp_x);
        y.push_back(tmp_y);
    }

    fin.close();
}

template<typename A, typename B, typename C>
void outputParams(string dir, neuralnet<A, B, C>& net) {
    fstream fout(dir, ios::out);
    fout.precision(10);

    for(int i = 0; i < net.getWeights().size(); i++) {
        fout << (*(net.getWeights()[i])) << "\n\n";
    }
    fout << "\n\n\n";
    for(int i = 1; i < net.getBiases().size(); i++) {
        fout << (*(net.getBiases()[i])) << "\n\n";
    }

    fout.close();
}

template<typename A, typename B, typename C>
void outputMapping(string dir, neuralnet<A, B, C>& net, const int& ins) {
    fstream fout(dir, ios::out);
    fout.precision(10);

    fout << 201 << "\n";
    for(double i = 0; i < 10.05; i += 0.05) {
        if(abs(i) < 10e-6)
            i = 0;
        Eigen::VectorXd eval_tmp(ins);
        double p = i;
        for(int j = 0; j < ins; j++) {
            eval_tmp.coeffRef(j) = p;
            p *= i;
        }
        net.feedforward(eval_tmp);

        fout << i << " " << net.getOutput().transpose() << "\n";
    }

    fout.close();
}

int main() {
    // data
    std::vector<Eigen::VectorXd> X;
    std::vector<Eigen::VectorXd> y;

    // read data
    int n, ins, outs;
    readTrainingData("../dump/dataset.txt", X, y, n, ins, outs);

    // initialize the network
    neuralnet<Linear, Linear, SSE> net;
    net.addLayer(ins);
    net.addLayer(4);
    net.addLayer(4);
    net.addLayer(4);
    net.addOutput(outs);

    net.blindInit();

    net.attach(X, y);


#ifdef DEBUGGING
    auto start = high_resolution_clock::now();
#endif

    // fit the dataset
    // net.fit_with_regularization<L1>(0.0005, 5000, 32);
    // net.fit_with_optimizer<MSGD>(0.0005, 10000, 32);
    net.fit(0.0001, 10000, 32);
    cout << "yea------\n";
    //////////////////

#ifdef DEBUGGING
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    DEBUG("Execution time: ");
    DEBUG(duration.count());
    DEBUG(" ms\n");
#endif
    
    // output the parameters
    outputParams("../dump/weights.txt", net);

    // output the x to y mapping
    outputMapping("../dump/x_to_y.txt", net, ins);
}