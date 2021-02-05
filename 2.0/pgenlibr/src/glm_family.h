#ifndef __GLM_FAMILY_H__
#define __GLM_FAMILY_H__
#include <stdint.h>
// This class contains the response and the glm
// family of the response
class Family {
    protected:
    double * offset;
    uint32_t no;
    public:
    // Given the predicted value, compute the residual, return the negative log-likelihood 
    virtual double get_residual(const double * eta, double * r) const = 0;
    virtual double get_value(const double * eta) const = 0;
    virtual ~Family();

    uint32_t get_no() const;
};


class Gaussian : public Family {
    private:
    double *y;

    public:
    Gaussian(const double * original_y, const double * user_offset, const uint32_t no);
    ~Gaussian();

    double get_residual(const double * eta, double * r) const;
    double get_value(const double * eta) const;
};

class Logistic : public Family {
    private:
    double *y; // maybe just use int or bool?

    public:
    Logistic(const double * original_y, const double * user_offset, const uint32_t no);
    ~Logistic();
    double get_residual(const double * eta, double * r) const;
    double get_value(const double * eta) const;

};

typedef struct d2 {
    double value;
    double eta_mean;
    d2(const double v, const double e) : value(v), eta_mean(e){}
} d2;

class Cox : public Family {
    private:
    uint32_t *rank_to_ind; // maps rank to observation index
    // status, offset are both in the original order
    // Normalization is done through status
    uint32_t *rankmin;
    uint32_t *rankmax;
    double *rskdenom;
    double * status;
    double *offset;

    public:
    Cox(const int *rank, const int* rankmin, const int *rankmax, const double * status,  const double * user_offset, const uint32_t no);
    ~Cox();
    double get_residual(const double * eta, double * r) const;
    double get_value(const double * eta) const;
    d2 get_value_and_eta_mean(const double * eta) const;

};

#endif