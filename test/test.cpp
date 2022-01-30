#include "neuralnet.h"
#include "act_func.h"
#include <iostream>
#include <string>
#include <string.h>
#include <fstream>
using namespace std;

int main() {
    fstream fin("dataset.txt", ios::in);
    fstream fout("weights.txt", ios::out);
    fout.precision(10);
    neuralnet<linear, sse> net;

    net.addLayer(4);
    net.addLayer(3);
    net.addOutput(1);

    net.initWeights();

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
        Eigen::VectorXd tmp_x(4);
        Eigen::VectorXd tmp_y(1);

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
    // cout << X[0] << "\n\n";
    // cout << (*net.getTrainX()[0]) << "\n";

    // net.feedforward(X[0]);
    // cout << (*net.getWeights()[0]) << "\n";
    // double rate, epoch;
    // cout << "learning rate: ";  cin >> rate;
    // cout << "epoch: ";          cin >> epoch;
    // 0.001 5000
    // cout << (*(net.getWeights()[0])) << "\n";

    net.fit(0.0001, 10000);
    cout << "yea------\n";

    for(int i = 0; i < net.getWeights().size(); i++) {
        fout << (*(net.getWeights()[i])) << "\n\n";
    }
    fout << "\n\n\n";
    for(int i = 1; i < net.getBiases().size(); i++) {
        fout << (*(net.getBiases()[i])) << "\n\n";
    }
    // cout << (*net.getLayers()[1]) << "\n";
    // cout << net.getOutput() << "\n";
}