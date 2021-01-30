#include "glm_family.h"
#include <cstdlib>
#include <math.h>

uint32_t Family::get_no() const {
    return no;
}
Family::~Family(){}

Gaussian::Gaussian(const double *original_y, const double *user_offset, const uint32_t no) {
    this->no = no;
    y = (double *)malloc(sizeof(double) * no);
    offset = nullptr;
    if (user_offset) {
        for (uint32_t i = 0; i < no; ++i) {
            y[i] = original_y[i] - user_offset[i];
        }
    } else {
        for(uint32_t i = 0; i < no; ++i){
            y[i] = original_y[i];
        }
    }
}

Gaussian::~Gaussian(){
    free(y);
}

double Gaussian::get_residual(const double * eta, double * r) const {
    double result = 0;
    for(uint32_t i = 0; i < no; ++i){
        r[i] = (eta[i] - y[i]) * (2.0/no);
        result += (eta[i] - y[i]) * (eta[i] - y[i]) /no;
    }
    return result;
}

double Gaussian::get_value(const double *eta) const {
    double result = 0;
    for(uint32_t i = 0; i < no; ++i){
        result += (eta[i] - y[i]) * (eta[i] - y[i]) /no;
    }
    return result;
}

Logistic::Logistic(const double * original_y, const double * user_offset, const uint32_t no) {
    this->no = no;
    y = (double *)malloc(sizeof(double) * no);
    for(uint32_t i = 0; i < no; ++i) {
        y[i] = original_y[i];
    }
    if (user_offset) {
        offset = (double *)malloc(sizeof(double) * no);
        for (uint32_t i = 0; i < no; ++i) {
            offset[i] = user_offset[i];
        }
    } else {
        offset = nullptr;
    }
}

Logistic::~Logistic(){
    free(y);
    if(offset){
        free(offset);
    }
}

double Logistic::get_residual(const double * eta, double * r) const {
    if(offset){
        double result = 0;
        for(uint32_t i = 0; i < no; ++i){
            double tmp = exp(eta[i] + offset[i]);
            r[i] = (y[i] - 1/(1 + tmp))/no;
            result += (y[i] * eta[i] + log(1 + (1/tmp)))/no;
        }
        return result;
    }
    double result = 0;
    for(uint32_t i = 0; i < no; ++i) {
        double tmp = exp(eta[i]);
        r[i] = (y[i] - 1/(1 + tmp))/no;
        result += (y[i] * eta[i] + log(1 + (1/tmp)))/no;
    }
    return result;
}

double Logistic::get_value(const double *eta) const {
    if(offset){
        double result = 0;
        for(uint32_t i = 0; i < no; ++i){
            double tmp = exp(eta[i] + offset[i]);
            result += (y[i] * eta[i] + log(1 + (1/tmp)))/no;
        }
        return result;
    }
    double result = 0;
    for(uint32_t i = 0; i < no; ++i) {
        double tmp = exp(eta[i]);
        result += (y[i] * eta[i] + log(1 + (1/tmp)))/no;
    }
    return result;
}