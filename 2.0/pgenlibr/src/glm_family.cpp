#include "glm_family.h"
#include <cstdlib>
#include <math.h>
#include <iostream>

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
            double tmp = exp(-eta[i] - offset[i]);
            r[i] = (1/(1 + tmp) - y[i])/no;
            result += (log(1 + (1/tmp)) - (y[i] * eta[i]) )/no;
        }
        return result;
    }
    double result = 0;
    for(uint32_t i = 0; i < no; ++i) {
        double tmp = exp(-eta[i]);
        r[i] = (1/(1 + tmp) - y[i])/no;
        result += (log(1 + (1/tmp)) - (y[i] * eta[i]))/no;
    }
    return result;
}

double Logistic::get_value(const double *eta) const {
    if(offset){
        double result = 0;
        for(uint32_t i = 0; i < no; ++i){
            double tmp = exp(eta[i] + offset[i]);
            result += (log(1 + tmp) - (y[i] * eta[i]) )/no;
        }
        return result;
    }
    double result = 0;
    for(uint32_t i = 0; i < no; ++i) {
        double tmp = exp(eta[i]);
        result += (log(1 + tmp)- (y[i] * eta[i]))/no;
    }
    return result;
}

Cox::Cox(const int *rank, const int* rankmin, const int *rankmax, const double * status, const double * user_offset, const uint32_t no) {
    this->no = no;
    this->rank_to_ind = (uint32_t*)malloc(sizeof(uint32_t) * no);
    this->rankmin = (uint32_t*)malloc(sizeof(uint32_t) * no);
    this->rankmax = (uint32_t*)malloc(sizeof(uint32_t) * no);
    this->rskdenom = (double*)malloc(sizeof(double) * no);
    this->status = (double*)malloc(sizeof(double) * no);

    for(uint32_t i = 0; i < no; ++i){
        this->rank_to_ind[i] = rank[i];
        this->rankmin[i] = rankmin[i];
        this->rankmax[i] = rankmax[i];
        this->status[i] = status[i];
    }

    if(user_offset){
        offset = (double*)malloc(sizeof(double) * no);
        for(uint32_t i = 0; i < no; ++i){
            offset[i] = user_offset[i];
        }
    } else {
        offset = nullptr;
    }

}

Cox::~Cox(){
    free(rank_to_ind);
    free(rankmin);
    free(rankmax);
    free(rskdenom);
    free(status);
    if(offset){
        free(offset);
    }
}


double Cox::get_residual(const double * eta, double * r) const {
    // get_value will modify rskdenom to the right thing
    d2 result = get_value_and_eta_mean(eta);
    // reuse rskdenom
    double cumu = 0;
    for(uint32_t i = 0; i < no; ++i) {
        uint32_t ind = rank_to_ind[i];
        cumu += status[ind] / rskdenom[i];
        rskdenom[i] = cumu;
    }

    // Adjust for ties
    for(uint32_t i = 0; i < no; ++i) {
        rskdenom[i] = rskdenom[rankmax[i]];
    }

    for(uint32_t i = 0; i < no; ++i) {
        r[i] = -status[i];
    }

    if(offset){
        for(uint32_t i = 0; i < no; ++i) {
            uint32_t ind = rank_to_ind[i];
            r[ind] += exp(eta[ind] + offset[ind] - result.eta_mean) * rskdenom[i];
        }
        return result.value;
    }

    for(uint32_t i = 0; i < no; ++i) {
        uint32_t ind = rank_to_ind[i];
        r[ind] += exp(eta[ind] - result.eta_mean) * rskdenom[i];
    }
    return result.value;
}

d2 Cox::get_value_and_eta_mean(const double * eta) const {
    double eta_mean = 0;
    for(uint32_t i = 0; i < no; ++i){
        eta_mean += eta[i];
    }
    if(offset){
        for(uint32_t i = 0; i < no; ++i) {
            eta_mean += offset[i];
        }
    }
    eta_mean /= no;

    // First compute rskdenom
    if(offset){
        double cumu = 0;
        for(uint32_t i = 0; i < no; ++i) {
            uint32_t reverse_ind = rank_to_ind[no - i - 1];
            cumu += exp(eta[reverse_ind] + offset[reverse_ind] - eta_mean);
            rskdenom[no-1-i] = cumu;
        }
    } else {
        double cumu = 0;
        for(uint32_t i = 0; i < no; ++i) {
            uint32_t reverse_ind = rank_to_ind[no - i - 1];
            cumu += exp(eta[reverse_ind] - eta_mean);
            rskdenom[no-1-i] = cumu;
        }
    }

    // std::cout << "rskdenom[0] is" << rskdenom[0] << std::endl; 
    // std::cout << "eta_mean is" << eta_mean << std::endl;


    // adjust for ties
    double result = 0;
    for(uint32_t i = 0; i < no; ++i) {
        rskdenom[i] = rskdenom[rankmin[i]];
        uint32_t ind = rank_to_ind[i];
        result += status[ind] * log(rskdenom[i]) - status[i] * (eta[i] - eta_mean);
    }
    //std::cout << "result is" << result << std::endl;

    d2 d2result(result, eta_mean);
    return d2result;
}

double Cox::get_value(const double * eta) const {
    d2 result = get_value_and_eta_mean(eta);
    return result.value;
}