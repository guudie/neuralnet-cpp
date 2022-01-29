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
    fout.precision(20);
    neuralnet<linear> net;

    net.addLayer(4);
    net.addOutput(1);

    net.initWeights();

    std::vector<Eigen::VectorXd> X;
    std::vector<Eigen::VectorXd> y;
    int n;

    fin >> n;
    string tmp;
    fin >> tmp >> tmp >> tmp >> tmp >> tmp;
    // fin.ignore();
    int place;
    double x, x2, x3, x4, y_i;
    for(int i = 0; i < n; i++) {
        fin >> place >> x >> x2 >> x3 >> x4 >> y_i;
        // cout << x << x2 << x3 << x4 << "\n";
        Eigen::VectorXd tmp_x(4);
        tmp_x << x, x2, x3, x4;
        Eigen::VectorXd tmp_y(1);
        tmp_y << y_i;

        X.push_back(tmp_x);
        y.push_back(tmp_y);
    }

    net.attach(X, y);
    // cout << X[0] << "\n\n";
    // cout << (*net.getTrainX()[0]) << "\n";

    // net.feedforward(X[0]);
    // cout << (*net.getLayers()[0]) << "\n";
    // cout << (*net.getWeights()[0]) << "\n";
    // cout << net.getOutput() << "\n";
    double rate, epoch;
    cout << "learning rate: ";  cin >> rate;
    cout << "epoch: ";          cin >> epoch;
    // 0.001 50

    net.fit(rate, epoch);
    cout << "yea------\n";

    for(int i = 0; i < net.getWeights().size(); i++) {
        fout << (*(net.getWeights()[i])) << "\n";
    }
    fout << "\n\n";
    for(int i = 1; i < net.getBiases().size(); i++) {
        fout << (*(net.getBiases()[i])) << "\n";
    }
}