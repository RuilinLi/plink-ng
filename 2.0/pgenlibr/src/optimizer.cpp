#include "optimizer.h"
#include <math.h>

double ProximalGradient::get_step_size(){
    return step_size;
}

ProximalGradient::ProximalGradient(const double *beta_init, const uint32_t ni){
    this->ni = ni;
    beta = (double *)malloc(sizeof(double) * ni);
    beta_next = (double *)malloc(sizeof(double) * ni);
    beta_prev = (double *)malloc(sizeof(double) * ni);
    grad = (double *)malloc(sizeof(double) * ni);
    step_size = 1;
    weight_old = 1;
    if(beta_init){
        for(uint32_t i = 0; i < ni; ++i){
            beta[i] = beta_init[i];
            beta_prev[i] = beta_init[i];

        }
    } else {
        for(uint32_t i = 0; i < ni; ++i){
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

void ProximalGradient::prox(double step_size, double lambda){
    // impelment prox operator later
    for(uint32_t i = 0; i < ni; ++i){
        beta_next[i] = beta[i] - step_size * grad[i];
    }
    return;
}

void ProximalGradient::nesterov_update(){
    weight_new = 0.5*(1+sqrt(1+4*weight_old*weight_old));
    for(uint32_t i = 0; i < ni; ++i) {
        beta[i] = beta_next[i] + ((weight_old - 1)/weight_new) * (beta_next[i] - beta_prev[i]);
        beta_prev[i] = beta_next[i];
    }
    weight_old = weight_new;

} 

double ProximalGradient::quadratic_diff() {
    double result = 0;
    for(uint32_t i = 0; i < ni; ++i) {
        result += grad[i] * (beta_next[i] - beta[i]) + (beta_next[i] - beta[i])  * (beta_next[i] - beta[i]) /(2 * step_size);
    }
    return result;
}

void solver(const sparse_snp& X, const Family *y, const double *beta_init, double * result_buffer)
{
    const uint32_t ni = X.Getncol();
    const uint32_t no = X.Getnrow();
    ProximalGradient prox(beta_init, ni);
    double *residual = (double *)malloc(sizeof(double)*no);
    double *eta = (double *)malloc(sizeof(double)*no);
    double current_val;
    double next_val;


    for(int t = 0; t < 2000; ++t){
        // Initialize eta, residual
        X.xv(prox.beta, eta);
        current_val = y->get_residual(eta, residual);

        // Compute gradient
        X.vtx(residual, prox.grad);
        // backtracking line search
        while(true){
            // Maybe try increasing step size occasionally?
            // merge gradient descent and prox together
            prox.prox(prox.step_size, 1.0);

            // compute the value at the proposed parameters prox.beta_next
            X.xv(prox.beta_next, eta);
            next_val = y->get_value(eta);
            double diff_bound = prox.quadratic_diff();

            if(next_val <= current_val + diff_bound){
                break;
            }
            prox.step_size /= 1.1;
        }

        //double val_change = abs(next_val - current_val)/fmax(1.0, abs(current_val));
        double val_change = abs((next_val - current_val)/current_val);
        if(val_change < 1e-5){
            Rprintf("Converged at iteration %d\n", t);
            break;
        }

        // Nesterov acceleration
        prox.nesterov_update();

        if(t % 100 == 0){
            Rprintf("value is %f, \n", next_val);
            Rprintf("step size is %f, \n", prox.step_size);
        }

    }
    // save the result
    for(uint32_t i = 0; i < ni; ++i){
        result_buffer[i] = prox.beta_next[i];
    }

    free(residual);
    free(eta);

}



// [[Rcpp::export]]
NumericVector SparseTest123(List mat, NumericVector y) {
  if (strcmp_r_c(mat[0], "sparse_snp")) {
    stop("matrix not the right type");
  }
  XPtr<class sparse_snp> x = as<XPtr<class sparse_snp> >(mat[1]);
  NumericVector result(x->Getncol());
  Gaussian response(&y[0], nullptr, x->Getnrow());
  solver(*x, &response, nullptr, &result[0]);
  return result;
}