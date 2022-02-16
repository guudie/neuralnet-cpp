#ifndef LAYER_H
#define LAYER_H

#ifndef NEURALNET_H
#include <eigen3/Eigen/Core>
#endif

class layer {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

    const int size_in;
    const int size_out;

public:
    layer(const int& ins, const int& outs) : size_in(ins), size_out(outs) {}
    virtual ~layer() {}

    
};

#endif