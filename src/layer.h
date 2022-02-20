#ifndef LAYER_H
#define LAYER_H

#include <eigen3/Eigen/Core>
#include <algorithm>
#include <random>
#include "optimizers.h"

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

    // initialize layer;
    virtual void init(const int& batch_size = 1) = 0;

    // random init
    virtual void randInit(const int& batch_size = 1) = 0;

    // calculate this layer's outputs
    virtual void evaluate(const mat& data_in) = 0;

    /** backprop algorithm
    * 
    * assumptions: current layer's [∂E / ∂a] has been calculated
    * to be computed: lower layer's [∂E / ∂a]
    *                 this layer's [∂E / ∂z]
    *                 this layer's [∂E / ∂W]
    *                 this layer's [∂E / ∂b]
    */
    virtual void backprop(const mat& lower_a, mat& lower_da) = 0;
    virtual void backprop(const mat& lower_a) = 0;  // same as above, but for the first hidden layer

    // update the parameters according to gradients
    // virtual void updateParams(const double& rate) = 0;
    virtual void updateParams(optimizer*& opt) = 0;
};

#endif