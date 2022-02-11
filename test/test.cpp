#define DEBUGGING

#ifdef DEBUGGING
#include <chrono>
using namespace std::chrono;
#endif

#include "inc.h"
#include <iostream>
#include <string>
#include <fstream>
using namespace std;

int main() {
    fstream fin("../dump/dataset.txt", ios::in);
    fstream fout("../dump/weights.txt", ios::out);
    fstream eval("../dump/x_to_y.txt", ios::out);
    fout.precision(10);
    eval.precision(10);
    neuralnet<linear, linear, sse> net;

    net.addLayer(2);
    net.addLayer(4);
    net.addLayer(4);
    net.addLayer(4);
    net.addOutput(1);

    net.init();

    std::vector<Eigen::VectorXd> X;
    std::vector<Eigen::VectorXd> y;

    int n, ins, outs;
    fin >> n >> ins >> outs;
    string tmp;
    fin.ignore();
    getline(fin, tmp);
    int place;
    double* x_i = new double[ins];
    double* y_i = new double[outs];
    for(int i = 0; i < n; i++) {
        Eigen::VectorXd tmp_x(ins);
        Eigen::VectorXd tmp_y(outs);

        fin >> place;
        for(int i = 0; i < ins; i++) {
            fin >> x_i[i];
            tmp_x.coeffRef(i) = x_i[i];
        }
        for(int i = 0; i < outs; i++) {
            fin >> y_i[i];
            tmp_y.coeffRef(i) = y_i[i];
        }

        X.push_back(tmp_x);
        y.push_back(tmp_y);
    }

    net.attach(X, y);


#ifdef DEBUGGING
    auto start = high_resolution_clock::now();
#endif

    // fit the dataset
    // net.fit_with_regularization<L1>(0.0005, 5000, 32);
    net.fit(0.0001, 10000, 32);
    cout << "yea------\n";
    //////////////////

#ifdef DEBUGGING
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Execution time: " << duration.count() << " ms\n";
#endif

    // output the weights
    for(int i = 0; i < net.getWeights().size(); i++) {
        fout << (*(net.getWeights()[i])) << "\n\n";
    }
    fout << "\n\n\n";
    for(int i = 1; i < net.getBiases().size(); i++) {
        fout << (*(net.getBiases()[i])) << "\n\n";
    }
    
    // output the x to y mapping
    eval << 201 << "\n";
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

        eval << i << " " << net.getOutput() << "\n";
    }
}