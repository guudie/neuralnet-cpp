#ifndef MSE_H
#define MSE_H

class MSE {
private:
    typedef Eigen::MatrixXd mat;
    typedef Eigen::VectorXd vec;
public:
    MSE();
    ~MSE();

    // compute ∂E / ∂ȳ
    static void diff(mat& dy, const mat& ybar, const mat& y) {
        double num = 2.0 / y.cols();
        dy.noalias() = (ybar - y) * num;
    }
};

#endif