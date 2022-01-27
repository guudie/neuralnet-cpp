
class relu {
public:
    relu();
    ~relu();

    double operator()(double x) {
        if(x < 0)
            return 0;
        if(x <= 1)
            return x;
        return 1;
    }

    static double diff(double x) {
        if(0 <= x && x <= 1)
            return 1;
        return 0;
    }
};