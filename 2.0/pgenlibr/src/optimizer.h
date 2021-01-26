#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include <stdint.h>
#include "glm_family.h"
#include "pgenlibr.h"

// This should really be called
// a proximal gradient method for group lasso
class ProximalGradient {
   private:
    // number of variables
    uint32_t ni;
    double weight_old;
    double weight_new;
    const int * group_cumu;
    const uint32_t ngroup;

   public:
    double* beta;
    double* grad;
    double* beta_next;
    double* beta_prev;
    double step_size;
    ProximalGradient(const double *beta_init, const uint32_t ni, const IntegerVector group);
    ~ProximalGradient();
    double get_step_size();
    // beta_next = prox(beta - step_size * grad)
    void prox(double lambda);
    // returns <grad, beta_next - beta> + \|beta_next - beta\|_2^2/(2*step_size)
    double quadratic_diff();
    void nesterov_update();
    void reset_nesterov_weight();
};

void solver(const sparse_snp &X, const Family *y, const double *beta_init);

#endif