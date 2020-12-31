#include "glm_family.h"
#include <cstdlib>
#include <math.h>

uint32_t Family::get_no(){
    return no;
}
Family::~Family(){}

Gaussian::Gaussian(const double *original_y, const double *user_offset) {
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

void Gaussian::get_residual(const double * eta, double * r){
    for(uint32_t i = 0; i < no; ++i){
        r[i] = y[i] - eta[i];
    }
}

Logistic::Logistic(const double * original_y, const double * user_offset) {
    y = (double *)malloc(sizeof(double) * no);
    for(uint32_t i = 0; i < no; ++i) {
        y[i] = original_y[i];
    }
    if (offset) {
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

void Logistic::get_residual(const double * eta, double * r){
    if(offset){
        for(uint32_t i = 0; i < no; ++i){
            r[i] = y[i] - 1/(1 + exp(eta[i] + offset[i]));
        }
        return;
    }
    for(uint32_t i = 0; i < no; ++i) {
        r[i] = y[i] - 1/(1 + exp(eta[i]));
    }

}