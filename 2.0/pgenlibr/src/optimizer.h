#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include <stdint.h>
#include "glm_family.h"
#include "pgenlibr.h"

// This should really be called
// a proximal gradient method for group lasso
class ProximalGradient {
   protected:
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
    virtual ~ProximalGradient();
    double get_step_size();
    // beta_next = prox(beta - step_size * grad)
    virtual void prox(double lambda); // The base class prox does nothing (no regularization)
    // returns <grad, beta_next - beta> + \|beta_next - beta\|_2^2/(2*step_size)
    double quadratic_diff();
    void nesterov_update();
    void reset_nesterov_weight();
    uint32_t get_ni();
};

class GroupPenalty : public ProximalGradient {
    private:
      uint32_t * group_cumu;
      const uint32_t ngroup;
    public:
      GroupPenalty(const double *beta_init, const uint32_t ni, const IntegerVector group);
      virtual void prox(double lambda);
      virtual ~GroupPenalty();
};

class LassoPenalty : public ProximalGradient {
    private:
      double * pfac;
    public:
      LassoPenalty(const double *beta_init, const uint32_t ni, Nullable<NumericVector> penalty_factor=R_NilValue);
      virtual void prox(double lambda);
      virtual ~LassoPenalty();
};

void solver(const sparse_snp &X, const Family *y, const double *beta_init);

#endif