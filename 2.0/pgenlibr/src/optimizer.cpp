#include "optimizer.h"

double ProximalGradient::get_step_size(){
    return step_size;
}

ProximalGradient::ProximalGradient(const double *beta_init, const uint32_t ni){
    this->ni = ni;
    beta = (double *)malloc(sizeof(double) * ni);
    step_size = 0.00001;
    if(beta_init){
        for(uint32_t i = 0; i < ni; ++i){
            beta[i] = beta_init[i];
        }
    } else {
        for(uint32_t i = 0; i < ni; ++i){
            beta[i] = 0;
        }
    }
}

ProximalGradient::~ProximalGradient(){
    free(beta);
}

void ProximalGradient::prox(double lambda){
    return;
}

void solver(const sparse_snp& X, const Family *y, const double *beta_init)
{
    const uint32_t ni = X.Getncol();
    const uint32_t no = X.Getnrow();
    ProximalGradient prox(beta_init, ni);
    double *residual = (double *)malloc(sizeof(double)*no);
    double *eta = (double *)malloc(sizeof(double)*no);
    double step_size_here = 1.0/(2.0*no);

    for(int t = 0; t < 10000; ++t){
        //Compute eta
        X.xv(prox.beta, eta);

        // Compute residual
        y->get_residual(eta, residual);

        // gradient descent
        X.vtx(residual, prox.beta, step_size_here);
        
        // prox operator, implement later
        prox.prox(1);

        if(t % 100 == 0){
            Rprintf("Iteration %d, beta[1] is %f\n", t, prox.beta[0]);
        }

    }

    free(residual);
    free(eta);

}



// [[Rcpp::export]]
void SparseTest123(List mat, NumericVector y) {
  if (strcmp_r_c(mat[0], "sparse_snp")) {
    stop("matrix not the right type");
  }
  XPtr<class sparse_snp> x = as<XPtr<class sparse_snp> >(mat[1]);
  Gaussian response(&y[0], nullptr, x->Getnrow());
  solver(*x, &response, nullptr);
}