#ifndef LAYER_H
#define LAYER_H

#ifndef NEURALNET_H
#include <eigen3/Eigen/Core>
#endif

#include <algorithm>
#include <random>

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
    virtual const mat& getData() const = 0;

    // reference to ∂E / ∂a
    virtual mat& daRef() = 0;

    // reference to ∂E / ∂z
    virtual mat& dzRef() = 0;

    // reference to ∂a / ∂z
    virtual mat& dazRef() = 0;

    // initialize layer;
    virtual void init(const int& batch_size = 1) = 0;

    // random init
    virtual void randInit(const int& batch_size = 1) = 0;

    // calculate this layer's outputs
    virtual void evaluate(const mat& data_in) = 0;

    // backprop algorithm
    //
    // assumptions: current layer's [∂E / ∂a], [∂E / ∂z] have been calculated
    // to be computed: lower layer's [∂E / ∂a]
    //                 this layer's [∂E / ∂W]
    //                 this layer's [∂E / ∂b]
    virtual void backprop(const layer* upper_layer, layer* lower_layer) = 0;
};

#endif