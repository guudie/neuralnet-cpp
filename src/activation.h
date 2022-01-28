
class relu {
public:
    relu();
    ~relu();

    double operator()(double x) {
        if(x < 0)
            return 0;
        return x;
    }

    static double diff(double x) {
        if(x <= 0)
            return 0;
        return 1;
    }
};