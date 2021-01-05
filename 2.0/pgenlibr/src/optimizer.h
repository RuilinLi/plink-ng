#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include <stdint.h>
#include "glm_family.h"
#include "pgenlibr.h"


class ProximalGradient {
   private:
    // number of variables
    uint32_t ni;
    double weight_old;
    double weight_new;

   public:
    double* beta;
    double* grad;
    double* beta_next;
    double* beta_prev;
    double step_size;
    ProximalGradient(const double *beta_init, const uint32_t ni);
    ~ProximalGradient();
    double get_step_size();
    // beta_next = prox(beta - step_size * grad)
    void prox(double step_size, double lambda);
    // returns <grad, beta_next - beta> + \|beta_next - beta\|_2^2/(2*step_size)
    double quadratic_diff();
    void nesterov_update();
};

void solver(const sparse_snp &X, const Family *y, const double *beta_init);

#endif