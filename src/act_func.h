#ifndef ACT_FUNC_H
#define ACT_FUNC_H

class linear {
public:
    linear() {}
    ~linear() {}

    static double f(double x) {
        return x;
    }

    static double diff(double) {
        return 1;
    }
};

class relu {
public:
    relu() {}
    ~relu() {}

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

#endif