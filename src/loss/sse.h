#ifndef SSE_H
#define SSE_H

class SSE {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;
public:
    SSE();
    ~SSE();

    // compute ∂E / ∂ȳ
    static void diff(mat& dy, const mat& ybar, const mat& y) {
        dy.noalias() = (ybar - y) * 2;
    }
};

#endif