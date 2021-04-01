#include "optimizer.h"
#include <math.h>

double ProximalGradient::get_step_size(){
    return step_size;
}

uint32_t ProximalGradient::get_ni(){
  return ni;
}

ProximalGradient::ProximalGradient(const double *beta_init, const uint32_t ni) 
{
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

GroupPenalty::GroupPenalty(const double *beta_init, const uint32_t ni,
                           const IntegerVector group)
    : ProximalGradient(beta_init, ni), ngroup(group.size() - 1) {
  group_cumu = (uint32_t *)malloc(sizeof(uint32_t) * group.size());
  for (uint32_t i = 0; i < group.size(); ++i) {
    group_cumu[i] = group[i];
  }
}

GroupPenalty::~GroupPenalty(){
  free(group_cumu);
}

LassoPenalty::LassoPenalty(const double *beta_init, const uint32_t ni,
                           Nullable<NumericVector> penalty_factor)
    : ProximalGradient(beta_init, ni) {
  pfac = (double *)malloc(sizeof(double) * ni);
  if (penalty_factor.isNotNull()) {
    NumericVector p = as<NumericVector>(penalty_factor);
    if (p.size() != ni) {
      stop("Penalty factor size incorrect");
    }
    for (uint32_t i = 0; i < ni; ++i) {
      pfac[i] = p[i];
    }
    return;
  }
  for (uint32_t i = 0; i < ni; ++i) {
    pfac[i] = 1;
  }
}

LassoPenalty::~LassoPenalty(){
  free(pfac);
}


ProximalGradient::~ProximalGradient(){
    free(beta);
    free(beta_next);
    free(beta_prev);
    free(grad);
}

void ProximalGradient::prox(double lambda){
    for(uint32_t i = 0; i < ni; ++i){
      beta_next[i] = beta[i] - step_size * grad[i];
    }
    return;
}

void GroupPenalty::prox(double lambda){
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
    return;
}

void LassoPenalty::prox(double lambda){
  for(uint32_t i = 0; i < ni; ++i){
    //beta_next[i] = beta[i] - step_size * grad[i];
    double grad_step = beta[i] - step_size * grad[i];
    beta_next[i] = copysign(fmax(abs(grad_step) - step_size * lambda * pfac[i], 0), grad_step);
  }
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

void solver(const base_snp &X, const Family &y, ProximalGradient &prox,
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
      current_val = y.get_residual(eta, residual);

      // Compute gradient
      X.vtx(residual, prox.grad);
      // backtracking line search
      int last_time_increase_step_size = -1;
      while (true) {
        // Maybe try increasing step size occasionally?
        int counter = 0;
        // merge gradient descent and prox together
        prox.prox(lambda);

        // compute the value at the proposed parameters prox.beta_next
        X.xv(prox.beta_next, eta);
        next_val = y.get_value(eta);
        double diff_bound = prox.quadratic_diff();


        if (next_val <= current_val + diff_bound) {
          if(counter == 0 && (last_time_increase_step_size == t - 1)){
            prox.step_size *= 1.1;
            last_time_increase_step_size = t;
          }
          break;
        }
        counter++;
        prox.step_size /= 1.1;
      }

      // double val_change = abs(next_val - current_val)/fmax(1.0,
      // abs(current_val));
      double val_change = abs((next_val - current_val) / current_val);
      if ((val_change < 1e-5) || (next_val < 1e-6)) {
        Rprintf("val_change is %f\n", val_change);
        Rprintf("next_val is %f\n", next_val);
        Rprintf("Converged at iteration %d\n", t);
        Rprintf("step size is %f, \n", prox.step_size);
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
SEXP NewResponseObj(NumericVector y, String family, Nullable<NumericVector> offset=R_NilValue){
  Family * fptr = nullptr;
  if(strcmp_r_c(family, "gaussian") == 0){
    fptr = new Gaussian(&y[0], nullptr, y.size());
  } else if (strcmp_r_c(family, "binomial") == 0) {
    if(offset.isNotNull()){
      NumericVector offset_vector(offset);
      fptr = new Logistic(&y[0], &offset_vector[0], y.size());
    } else {
      fptr = new Logistic(&y[0], nullptr, y.size());
    }
  } else {
    stop("This function only supports gaussian and binomial family");
  }
  XPtr<class Family> fxptr(fptr, true);
  return List::create(_["class"] = "response", _["response"] = fxptr);
}

// [[Rcpp::export]]
SEXP NewCoxResponseObj(NumericVector status, IntegerVector order0, IntegerVector rankmin0, IntegerVector rankmax0, Nullable<NumericVector> offset=R_NilValue){
  Family * fptr = nullptr;
  double *offsetptr = nullptr;
  if(offset.isNotNull()){
    NumericVector offset_vector(offset);
    offsetptr = &offset_vector[0];
  }
  fptr = new Cox(&order0[0], &rankmin0[0], &rankmax0[0], &status[0], offsetptr, status.size());
  XPtr<class Family> fxptr(fptr, true);
  return List::create(_["class"] = "response", _["response"] = fxptr);
}

// [[Rcpp::export]]
SEXP NewProxObj(int ni, IntegerVector group){
  ProximalGradient * pgptr = new GroupPenalty(nullptr, ni, group);
  XPtr<class ProximalGradient> pgxprt(pgptr, true);
  return List::create(_["class"] = "prox", _["prox"] = pgxprt);
}

// [[Rcpp::export]]
SEXP NewLassoObj(int ni, Nullable<NumericVector> pfac=R_NilValue, Nullable<NumericVector> beta=R_NilValue){
  double *beta_init = nullptr;
  if(beta.isNotNull()){
    NumericVector beta_vec = as<NumericVector>(beta);
    if(beta_vec.size() != ni){
      stop("Size of beta does not match number of variables");
    }
    beta_init = &beta_vec[0];
  }

  ProximalGradient * pgptr = new LassoPenalty(beta_init, ni, pfac);
  XPtr<class ProximalGradient> pgxprt(pgptr, true);
  return List::create(_["class"] = "prox", _["prox"] = pgxprt);
}

// [[Rcpp::export]]
SEXP GradientDescentObj(int ni, Nullable<NumericVector> beta=R_NilValue){
  double *beta_init = nullptr;
  if(beta.isNotNull()){
    NumericVector beta_vec = as<NumericVector>(beta);
    if(beta_vec.size() != ni){
      stop("Size of beta does not match number of variables");
    }
    beta_init = &beta_vec[0];
  }

  ProximalGradient * pgptr = new ProximalGradient(beta_init, ni);
  XPtr<class ProximalGradient> pgxprt(pgptr, true);
  return List::create(_["class"] = "prox", _["prox"] = pgxprt);
}

// [[Rcpp::export]]
NumericMatrix FitProx(List mat, List prox, List response, NumericVector lambda_seq) {
  if (strcmp_r_c(mat[0], "sparse_snp") && strcmp_r_c(mat[0], "dense_snp")) {
    stop("matrix not the right type");
  }

  if (strcmp_r_c(prox[0], "prox")) {
    stop("Proximal operator not the right type");
  }

  if (strcmp_r_c(response[0], "response")) {
    stop("response not the right type");
  }

  XPtr<class base_snp> x = as<XPtr<class base_snp> >(mat[1]);
  XPtr<class ProximalGradient> proxptr = as<XPtr<class ProximalGradient> >(prox[1]);
  XPtr<class Family> responseptr = as<XPtr<class Family> >(response[1]);

  const uint32_t no = x->Getnrow();
  const uint32_t ni = x->Getncol();
  if(no != responseptr->get_no()){
    stop("The row number of the matrix does not match the length of y");
  }

  if(ni != proxptr->get_ni()){
    stop("The column number of the matrix does not match the proximal operator");
  }

  NumericMatrix result(ni, lambda_seq.size());
  solver(*x, *responseptr, *proxptr, lambda_seq, &result[0]);
  return result;
}


// [[Rcpp::export]]
double ComputeLambdaMax(List mat, List response, IntegerVector group) {
  if (strcmp_r_c(mat[0], "sparse_snp") && strcmp_r_c(mat[0], "dense_snp")) {
    stop("matrix not the right type");
  }

  if (strcmp_r_c(response[0], "response")) {
    stop("response not the right type");
  }

  XPtr<class base_snp> x = as<XPtr<class base_snp> >(mat[1]);
  XPtr<class Family> responseptr = as<XPtr<class Family> >(response[1]);
  const uint32_t no = x->Getnrow();
  const uint32_t ni = x->Getncol();

  if(no != responseptr->get_no()){
    stop("The row number of the matrix does not match the length of y");
  }

  double *residual = (double *)malloc(sizeof(double)*no);
  double *eta = (double *)malloc(sizeof(double)*no);
  double *grad = (double *)malloc(sizeof(double)*(ni));

  for(uint32_t i = 0; i < no; ++i){
      eta[i] = 0;
  }
  responseptr->get_residual(eta, residual);
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
  // NumericVector toreturn(1);
  // toreturn[0] = result;
  free(residual);
  free(eta);
  free(grad);
  return result;
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