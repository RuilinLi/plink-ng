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
    // Given the predicted value, compute the residual
    virtual void get_residual(const double * eta, double * r) const = 0;
    virtual ~Family();

    uint32_t get_no() const;
};


class Gaussian : public Family {
    private:
    double *y;

    public:
    Gaussian(const double * original_y, const double * user_offset, const uint32_t no);
    ~Gaussian();

    void get_residual(const double * eta, double * r) const;
};

class Logistic : public Family {
    private:
    double *y; // maybe just use int or bool?

    public:
    Logistic(const double * original_y, const double * user_offset, const uint32_t no);
    ~Logistic();
    void get_residual(const double * eta, double * r) const;

};

#endif