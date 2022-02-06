#ifndef REGURIZER_H
#define REGURIZER_H

class L1 {
public:
    L1();
    ~L1();

    static double diff(const double& w) {
        if(w < 0)
            return -1;
        return 1;
    }
};

class L2 {
public:
    L2();
    ~L2();

    static double diff(const double& w) {
        return 2 * w;
    }
};

#endif