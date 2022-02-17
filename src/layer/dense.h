#ifndef DENSE_H
#define DENSE_H

#ifndef LAYER_H
#include "../layer.h"
#endif

// defining dense layer, fully connected to all neurons from previous layer

// the terms are matrices, each column is an input, rows represent the neurons

template<typename activation>
class dense : public layer {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;

    mat* a;         // activated terms: σ(z)  (size_out x obs)
    mat* z;         // calculated terms: z = W * a[i-1] + b  (size_out x obs)
    mat* da;        // ∂E / ∂a
    mat* dz;        // ∂E / ∂z = [∂E / ∂a] * [∂a / ∂z]
    mat* din;       // ∂a / ∂z = σ'(z)

    vec* b;         // bias of this layer   (size_out x 1)
    vec* db;        // ∂E / ∂b  (size_out x 1)

    mat* W;         // weights  (size_out x size_in)
    mat* dW;        // ∂E / ∂W  (size_out x size_in)

public:
    dense(const int& ins, const int& outs) : layer(ins, outs) {
        a = NULL;
        z = NULL;
        da = NULL;
        dz = NULL;
        din = NULL;
        b = NULL;
        db = NULL;
        W = NULL;
        dW = NULL;
    }
    ~dense() {
        delete a;
        delete z;
        delete da;
        delete dz;
        delete b;
        delete db;
    }

    // get output data
    const mat& getData() {
        return *a;
    }

    // initialize all params
    void init() {
        // tf is this?? delete it
        a = new mat(size_out, 1);
        z = new mat(size_out, 1);
        b = new vec(size_out);
        W = new mat(size_out, size_in);
        W->array() = 1;
        b->array() = 2;
    }

    // calculate this layer's terms and activate them
    void evaluate(const mat& data_in) {
        z->noalias() = (*W) * data_in;
        z->colwise() += *b;
        activation::f(*a, *z);
    }
};

#endif