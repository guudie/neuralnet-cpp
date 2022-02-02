#ifndef ACT_FUNC_H
#define ACT_FUNC_H

class linear {
public:
    linear();
    ~linear();

    static double f(double x) {
        return x;
    }

    static double diff(double) {
        return 1;
    }
};

class relu {
public:
    relu();
    ~relu();

    static double f(double x) {
        if(x < 0)
            return 0;
        return x;
    }

    static double diff(double x) {
        if(x < 0)
            return 0;
        return 1;
    }
};

class sigmoid {
public:
    sigmoid();
    ~sigmoid();

    static double f(double x) {
        return 1 / (1 + exp(-x));
    }

    static double diff(double x) {
        double tmp = f(x);
        return tmp - tmp*tmp;
    }
};

class fastmoid {
public:
    fastmoid();
    ~fastmoid();

    static double f(double x) {
        return (x / (1 + abs(x)) + 1) / 2;
    }

    static double diff(double x) {
        double tmp = 1 + abs(x);
        return 1/tmp/tmp/2;
    }
};

class softmax {
public:
    softmax();
    ~softmax();

    
};

#endif