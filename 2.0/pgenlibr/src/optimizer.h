#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include <stdint.h>
#include "glm_family.h"
#include "pgenlibr.h"


class ProximalGradient {
   private:
    // number of variables
    uint32_t ni;
    double step_size;

   public:
    double* beta;
    ProximalGradient(const double *beta_init, const uint32_t ni);
    ~ProximalGradient();
    double get_step_size();
    void prox(double lambda);
};

void solver(const sparse_snp &X, const Family *y, const double *beta_init);

#endif