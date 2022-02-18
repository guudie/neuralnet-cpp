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
    mat* daz;       // ∂a / ∂z = σ'(z)

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
        daz = NULL;
        b = NULL;
        db = NULL;
        W = NULL;
        dW = NULL;
    }
    ~dense() {
        clearAll();
    }

    // clear
    void clearAll() {
        delete a;
        delete z;
        delete da;
        delete dz;
        delete daz;
        delete b;
        delete db;
        delete W;
        delete dW;
    }

    // get output data
    const mat& getData() const {
        return *a;
    }

    // reference to ∂E / ∂a
    mat& daRef() {
        return *da;
    }

    // reference to ∂E / ∂z
    mat& dzRef() {
        return *dz;
    }

    // reference to ∂a / ∂z
    mat& dazRef() {
        return *daz;
    }

    // initialize all params
    void init(const int& batch_size = 1) {
        clearAll();
        a       = new mat(size_out, batch_size);
        z       = new mat(size_out, batch_size);
        da      = new mat(size_out, batch_size);
        dz      = new mat(size_out, batch_size);
        daz     = new mat(size_out, batch_size);
        b       = new vec(size_out);
        db      = new vec(size_out);
        W       = new mat(size_out, size_in);
        dW      = new mat(size_out, size_in);
    }

    // random init
    void randInit(const int& batch_size = 1) {
        init();
        std::random_device r;
        std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
        std::mt19937 eng(seed);
        std::uniform_real_distribution<double> urd(-.5, .5);

        for(int i = 0; i < W->rows(); i++) {
            for(int j = 0; j < W->cols(); j++)
                W->coeffRef(i, j) = urd(eng);
            b->coeffRef(i) = urd(eng);
        }
    }

    // calculate this layer's terms and activate them
    void evaluate(const mat& data_in) {
        z->noalias() = (*W) * data_in;
        z->colwise() += *b;
        activation::f(*a, *z);
        // compute [∂a / ∂z]
        activation::diff(*daz, *z, *a);
    }

    // backprop algorithm
    //
    // assumptions: current layer's [∂E / ∂a], [∂E / ∂z] have been calculated
    // to be computed: lower layer's [∂E / ∂a]
    //                 this layer's [∂E / ∂W]
    //                 this layer's [∂E / ∂b]
    void backprop(const layer* upper_layer, layer* lower_layer) {
        const double size = a->cols();
        // compute lower layer's [∂E / ∂a]
        lower_layer->daRef().noalias() = W->transpose() * (*dz);
        // compute lower layer's [∂E / ∂z]
        lower_layer->dzRef().noalias() = lower_layer->daRef().cwiseProduct(lower_layer->dazRef());
        // computer current layer's [∂E / ∂W]
        dW->noalias() = (*dz) * lower_layer->getData().transpose() / size;
        // computer current layer's [∂E / ∂b]
        db->noalias() = dz->rowwise().mean();
    }
};

#endif