#ifndef LAYER_H
#define LAYER_H

#ifndef NEURALNET_H
#include <eigen3/Eigen/Core>
#endif

class layer {
protected:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

    const int size_in;
    const int size_out;

public:
    layer(const int& ins, const int& outs) : size_in(ins), size_out(outs) {}
    virtual ~layer() {}

    // get output data
    virtual const mat& getData() = 0;

    // initialize layer;
    virtual void init() = 0;

    // calculate this layer's outputs
    virtual void evaluate(const mat& data_in) = 0;
};

#endif