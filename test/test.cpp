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

template<typename loss>
void outputMapping(string dir, neuralnet<loss>& net, int ins) {
    fstream fout(dir, ios::out);
    fout << 201 << "\n";
    vec tmp(ins);
    int L = net.size()-1;
    for(double i = 0; i < 10.05; i+=0.05) {
        tmp.coeffRef(0) = i;
        for(int j = 1; j < ins; j++)
            tmp.coeffRef(j) = tmp.coeff(j-1) * i;
        net.feedforward(tmp);
        fout << i << " " << net[L].getData().transpose() << "\n";
    }
}

int main() {
    vecContainer X;
    vecContainer y;
    int ins, outs;
    fetchData("../dump/dataset.txt", X, y, ins, outs);
    // cout << ins << " " << outs;

    neuralnet<SSE> net(500, 32);
    // optimizer* opt = new SGD(0.0001);
    // optimizer* opt = new MSGD(0.00005, 0.9);
    // optimizer* opt = new AdaGrad(0.1);
    optimizer* opt = new AdaDelta(0.01, 0.9);
    net.addLayer(new dense<Linear>(ins, 4));
    net.addLayer(new dense<Linear>(4, 4));
    net.addLayer(new dense<Linear>(4, 4));
    net.addLayer(new dense<Linear>(4, 4));
    net.addLayer(new dense<Linear>(4, outs));
    net.randInit();
    net.attach(X, y);
    
#ifdef DEBUGGING
    auto start = high_resolution_clock::now();
#endif

    net.fit(opt);
    
#ifdef DEBUGGING
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    debug("Execution time: ");
    debug(duration.count());
    debug(" ms\n");
#endif

    outputMapping("../dump/x_to_y.txt", net, ins);
}