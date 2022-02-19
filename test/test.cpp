#include "log.h"

#include "inc.h"
#include <iostream>
#include <fstream>
using namespace std;

typedef Eigen::VectorXd vec;
typedef std::vector<vec*> vecContainer;

void fetchData(string dir, vecContainer& X, vecContainer& y, int& ins, int& outs) {
    fstream fin(dir, ios::in);
    int n;
    fin >> n >> ins >> outs;
    for(int i = 0; i < n; i++) {
        double get;
        vec* tmp_i = new vec(ins);
        vec* tmp_o = new vec(outs);
        for(int j = 0; j < ins; j++) {
            fin >> get;
            tmp_i->coeffRef(j) = get;
        }
        for(int j = 0; j < outs; j++) {
            fin >> get;
            tmp_o->coeffRef(j) = get;
        }
        X.push_back(tmp_i);
        y.push_back(tmp_o);
    }
}


int main() {
    vecContainer X;
    vecContainer y;
    int ins, outs;
    fetchData("../dump/dataset.txt", X, y, ins, outs);
    // cout << ins << " " << outs;

    neuralnet<SSE> net(0.0001, 10000, 32);
    net.addLayer(new dense<Linear>(ins, 4));
    net.addLayer(new dense<Linear>(4, 4));
    net.addLayer(new dense<Linear>(4, 4));
    net.addLayer(new dense<Linear>(4, 4));
    net.addLayer(new dense<Linear>(4, outs));
    net.randInit();
    net.attach(X, y);
    // debug("yea");
    net.fit();
    // debug("yea");

    vec test(1);
    test << 10;

    net.feedforward(test);
    cout << net.getNetwork()[4]->getData();
}