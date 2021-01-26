#include "optimizer.h"
#include <math.h>

double ProximalGradient::get_step_size(){
    return step_size;
}

ProximalGradient::ProximalGradient(const double *beta_init, const uint32_t ni,
                                   const IntegerVector group)
    : group_cumu(&group[0]), ngroup(group.size() - 1) {
  this->ni = ni;
  beta = (double *)malloc(sizeof(double) * ni);
  beta_next = (double *)malloc(sizeof(double) * ni);
  beta_prev = (double *)malloc(sizeof(double) * ni);
  grad = (double *)malloc(sizeof(double) * ni);
  step_size = 1;
  weight_old = 1;
  if (beta_init) {
    for (uint32_t i = 0; i < ni; ++i) {
      beta[i] = beta_init[i];
      beta_prev[i] = beta_init[i];
    }
  } else {
    for (uint32_t i = 0; i < ni; ++i) {
      beta[i] = 0;
      beta_prev[i] = 0;
    }
  }
}

ProximalGradient::~ProximalGradient(){
    free(beta);
    free(beta_next);
    free(beta_prev);
    free(grad);
}

void ProximalGradient::prox(double lambda){
    // impelment prox operator later
    #pragma omp parallel for
    for(uint32_t g = 0; g < ngroup; ++g){
        double gnorm = 0;
        uint32_t start = group_cumu[g];
        uint32_t end = group_cumu[g+1];
        uint32_t gsize = end - start;
        for(uint32_t i = start; i < end; ++i){
            beta_next[i] = beta[i] - step_size * grad[i];
            gnorm += beta_next[i] * beta_next[i];
        }
        gnorm = sqrt(gnorm);
        double multiplier = 0;
        // The actual regularization term is lambda * sqrt(gsize)
        if(gnorm > (step_size * lambda * sqrt(gsize))) {
            multiplier = 1 - ((step_size * lambda * sqrt(gsize))/gnorm);
        }
        for(uint32_t i = start; i < end; ++i){
            beta_next[i] *= multiplier;
        }
    }
    // for(uint32_t i = 0; i < ni; ++i){
    //     beta_next[i] = beta[i] - step_size * grad[i];
    // }
    return;
}

void ProximalGradient::nesterov_update(){
    weight_new = 0.5*(1+sqrt(1+4*weight_old*weight_old));
    #pragma omp parallel for
    for(uint32_t i = 0; i < ni; ++i) {
        beta[i] = beta_next[i] + ((weight_old - 1)/weight_new) * (beta_next[i] - beta_prev[i]);
        beta_prev[i] = beta_next[i];
    }
    weight_old = weight_new;

} 

double ProximalGradient::quadratic_diff() {
    double result = 0;
    #pragma omp parallel for reduction(+:result)
    for(uint32_t i = 0; i < ni; ++i) {
        result += grad[i] * (beta_next[i] - beta[i]) + (beta_next[i] - beta[i])  * (beta_next[i] - beta[i]) /(2 * step_size);
    }
    return result;
}

void ProximalGradient::reset_nesterov_weight(){
    weight_old = 1;
}

void solver(const sparse_snp &X, const Family *y, ProximalGradient &prox,
            NumericVector lambda_seq, double *result_buffer) {
  const uint32_t ni = X.Getncol();
  const uint32_t no = X.Getnrow();
  // ProximalGradient prox(beta_init, ni);
  double *residual = (double *)malloc(sizeof(double) * no);
  double *eta = (double *)malloc(sizeof(double) * no);
  double current_val;
  double next_val;

  for (uint32_t lamind = 0; lamind < lambda_seq.size(); ++lamind) {
    double lambda = lambda_seq[lamind];

    for (int t = 0; t < 2000; ++t) {
      // Initialize eta, residual
      X.xv(prox.beta, eta);
      current_val = y->get_residual(eta, residual);

      // Compute gradient
      X.vtx(residual, prox.grad);
      // backtracking line search
      while (true) {
        // Maybe try increasing step size occasionally?
        // merge gradient descent and prox together
        prox.prox(lambda);

        // compute the value at the proposed parameters prox.beta_next
        X.xv(prox.beta_next, eta);
        next_val = y->get_value(eta);
        double diff_bound = prox.quadratic_diff();

        if (next_val <= current_val + diff_bound) {
          break;
        }
        prox.step_size /= 1.1;
      }

      // double val_change = abs(next_val - current_val)/fmax(1.0,
      // abs(current_val));
      double val_change = abs((next_val - current_val) / current_val);
      if ((val_change < 1e-5) || (next_val < 1e-6)) {
        Rprintf("val_change is %f\n", val_change);
        Rprintf("next_val is %f\n", next_val);
        Rprintf("Converged at iteration %d\n", t);
        break;
      }

      // Nesterov acceleration
      prox.nesterov_update();

      if (t % 10 == 0) {
        R_CheckUserInterrupt();
        Rprintf("value is %f, \n", next_val);
        Rprintf("step size is %f, \n", prox.step_size);
      }
    }
    // save the result
    for (uint32_t i = 0; i < ni; ++i) {
      result_buffer[lamind * ni + i] = prox.beta_next[i];
    }
    Rprintf("Results for Lambda index %d obtained\n", lamind+1);
    //prox.reset_nesterov_weight();
  }

  free(residual);
  free(eta);
}



// [[Rcpp::export]]
NumericMatrix SparseTest123(List mat, NumericVector y, IntegerVector group, NumericVector lambda_seq) {
  if (strcmp_r_c(mat[0], "sparse_snp")) {
    stop("matrix not the right type");
  }
  XPtr<class sparse_snp> x = as<XPtr<class sparse_snp> >(mat[1]);
  NumericMatrix result(x->Getncol(), lambda_seq.size());
  Gaussian response(&y[0], nullptr, x->Getnrow());
  ProximalGradient prox(nullptr, x->Getncol(), group);
  solver(*x, &response, prox, lambda_seq, &result[0]);
  return result;
}

// [[Rcpp::export]]
NumericVector ComputeLambdaMax(List mat, NumericVector y, IntegerVector group,
                               NumericVector offset) {
  if (strcmp_r_c(mat[0], "sparse_snp")) {
    stop("matrix not the right type");
  }
  XPtr<class sparse_snp> x = as<XPtr<class sparse_snp> >(mat[1]);
  uint32_t no = y.size();
  double *user_offset = &offset[0];
  Gaussian response(&y[0], user_offset, no);


  double *residual = (double *)malloc(sizeof(double)*no);
  double *eta = (double *)malloc(sizeof(double)*no);
  double *grad = (double *)malloc(sizeof(double)*(x->Getncol()));

  for(uint32_t i = 0; i < no; ++i){
      eta[i] = 0;
  }
  response.get_residual(eta, residual);
  x->vtx(residual, grad);
  double result = 0;
  for(uint32_t g = 0; g < (group.size() - 1); ++g){
    double gnorm = 0;
    uint32_t start = group[g];
    uint32_t end = group[g+1];
    uint32_t gsize = end - start;
    for(uint32_t i = start; i < end; ++i){
        gnorm += grad[i] * grad[i];
    }
    gnorm = sqrt(gnorm / gsize);
    result = fmax(result , gnorm);
  }
  NumericVector toreturn(1);
  toreturn[0] = result;
  free(residual);
  free(eta);
  free(grad);
  return toreturn;
}

// A utility function, might as well put here
// [[Rcpp::export]]
IntegerVector match_sorted_snp(IntegerVector chr, IntegerVector pos, IntegerVector refpos, IntegerVector refcumu) {
  uint32_t n = chr.size();
  IntegerVector result(n);
  for(uint32_t i = 0; i < n; ++i) {
      uint32_t current_chr = chr[i];
      uint32_t current_pos = pos[i];
      uint32_t start = refcumu[current_chr - 1];
      uint32_t end = refcumu[current_chr];
      // Binary search to find the index
      uint32_t max_iter = ceil(log2((double)(end - start + 1)));
      for(uint32_t j = 0; j < max_iter; ++j){
          uint32_t ind = (start + end)/2;
          if(refpos[ind] > current_pos){
              end = ind;
          } else if (refpos[ind] < current_pos) {
              start = ind;
          } else {
              result[i] = ind + 1;
              break;
          }
      }
  }
  return result;
}