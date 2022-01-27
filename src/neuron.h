
class neuron {
private:
    double val;
    double error;
public:
    neuron();
    ~neuron();

    double getVal() { return val; }
    void setVal(double _val) { val = _val; }

    double getError() { return error; }
    void setError(double _err) { error = _err; }
};