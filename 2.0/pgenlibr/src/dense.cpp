#include "pgenlibr.h"
#include "omp.h"

uint32_t base_snp::Getncol() const {
    return ni + ncov;
}

uint32_t base_snp::Getnrow() const {
    return no;
}

base_snp::~base_snp(){

}


void RPgenReader::LoadDense(dense_snp &x, IntegerVector variant_subset, Nullable<NumericMatrix> covariates) {
    if (!_info_ptr)
    {
        stop("pgen is closed");
    }
    uint32_t vsubset_size = variant_subset.size();
    x.no = _subset_size;
    x.ni = vsubset_size;

    x.xim = (double*)malloc(sizeof(double)*vsubset_size);

    ReadCompactListNoDosage(&x.genovec, variant_subset, x.xim);
    size_t genovec_cacheline_ct = plink2::DivUp(_subset_size, plink2::kNypsPerCacheline);
    x.genovec_word_ct = genovec_cacheline_ct * plink2::kWordsPerCacheline;

    if(covariates.isNotNull()){
        NumericMatrix covM = as<NumericMatrix>(covariates);
        if(covM.nrow() != _subset_size){
            stop("Incorrect covariates dimensions");
        }
        // Copy the covariates to cov
        x.ncov = covM.ncol();
        x.cov = (double*)malloc(sizeof(double) * covM.ncol() * covM.nrow());
        for(uint32_t i = 0; i < covM.ncol() * covM.nrow(); ++i){
            x.cov[i] = covM[i];
        }
    }
    x.loaded = true;
    return;
}

dense_snp::dense_snp(){
    no = 0;
    ni = 0;
    ncov = 0;
    cov = nullptr;
    loaded = false;
}

dense_snp::~dense_snp(){
    if (loaded) {
        free(xim);
        plink2::aligned_free(genovec);
        if(cov){
            free(cov);
        }
    }
}



void dense_snp::ResetMeanImputation(const NumericVector meanimp) {
    if(ni != meanimp.size()){
        stop("mean imputation does not have the right size");
    }
    for(uint32_t i = 0 ; i < ni; ++i){
        this->xim[i] = meanimp[i];
    }
}

void dense_snp::vtx(const double *v, double *result) const{
    if(!loaded){
        stop("matrix not loaded yet");
    }
    uint32_t local_word_ct = plink2::DivUp(no, plink2::kBitsPerWordD2);
    #pragma omp parallel for
    for(uint32_t densecol_ind = 0; densecol_ind < ni; ++densecol_ind){
        const uintptr_t* col = &(genovec[((uintptr_t)densecol_ind) * genovec_word_ct]);
        double result_1 = 0;
        double result_2 = 0;
        double result_missing = 0;
        for (uint32_t widx = 0; widx < local_word_ct; ++widx) {
            const uintptr_t geno_word = col[widx];
            const double *cur_weights = &(v[widx * plink2::kBitsPerWordD2]);
            uintptr_t geno_word1 = geno_word & plink2::kMask5555;
            uintptr_t geno_word2 = (geno_word >> 1) & plink2::kMask5555;
            uintptr_t geno_missing_word = geno_word1 & geno_word2;
            geno_word1 ^= geno_missing_word;
            while (geno_word1) {
                const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word1) / 2;
                result_1 +=  cur_weights[sample_idx_lowbits];
                geno_word1 &= geno_word1 - 1;
            }
            geno_word2 ^= geno_missing_word;
            while (geno_word2) {
                const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word2) / 2;
                result_2 +=  cur_weights[sample_idx_lowbits];
                geno_word2 &= geno_word2 - 1;
            }
            while (geno_missing_word) {
                const uint32_t sample_idx_lowbits = plink2::ctzw(geno_missing_word) / 2;
                result_missing +=  cur_weights[sample_idx_lowbits];
                geno_missing_word &= geno_missing_word - 1;
            }
        }
        result[densecol_ind] = (result_1 + 2 * result_2 + xim[densecol_ind] * result_missing);
    }

    #pragma omp parallel for
    for(uint32_t j = 0; j < ncov; ++j){
        double dot_prod = 0;
        for(uint32_t i = 0; i < no; ++i){
            dot_prod += cov[j*no + i] * v[i];
        }
        result[j + ni] = dot_prod;
    }

}

void dense_snp::xv(const double *v, double *result) const {
    if(!loaded){
        stop("matrix not loaded yet");
    }
    for(uint32_t i = 0; i < no; ++i){
        result[i] = 0;
    }
    const uint32_t word_ct_local = plink2::DivUp(no, plink2::kBitsPerWordD2);

#pragma omp parallel
    {
        uint32_t total_threads = omp_get_num_threads();
        uint32_t threadid = omp_get_thread_num();
        uint32_t size = (word_ct_local + total_threads * 8 - 1) / (total_threads * 8);
        size *= 8;
        uint32_t start = threadid * size;
        uint32_t end = (1 + threadid) * size;
        if (end > word_ct_local)
        {
            end = word_ct_local;
        }
        for (uint32_t j = 0; j < ni; ++j)
        {
            const uintptr_t *genoarr = &(genovec[((uintptr_t)j) * genovec_word_ct]);
            // it's easy to forget that xim should be extended when
            // we add covariates to it
            double ximpute = xim[j];
            double wj = v[j];

            for (uint32_t widx = start; widx < end; ++widx)
            {
                const uintptr_t geno_word = genoarr[widx];
                if (!geno_word)
                {
                    continue;
                }
                uintptr_t geno_word1 = geno_word & plink2::kMask5555;
                uintptr_t geno_word2 = (geno_word >> 1) & plink2::kMask5555;
                uintptr_t geno_missing_word = geno_word1 & geno_word2;
                geno_word1 ^= geno_missing_word;
                while (geno_word1)
                {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word1) / 2;
                    result[widx * plink2::kBitsPerWordD2 + sample_idx_lowbits] += wj;
                    geno_word1 &= geno_word1 - 1;
                }
                geno_word2 ^= geno_missing_word;
                while (geno_word2)
                {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word2) / 2;
                    result[widx * plink2::kBitsPerWordD2 + sample_idx_lowbits] += 2 * wj;
                    geno_word2 &= geno_word2 - 1;
                }
                while (geno_missing_word)
                {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_missing_word) / 2;
                    result[widx * plink2::kBitsPerWordD2 + sample_idx_lowbits] += ximpute * wj;
                    geno_missing_word &= geno_missing_word - 1;
                }
            }
        }
        start *= plink2::kBitsPerWordD2;
        end *= plink2::kBitsPerWordD2;
        if(end > no){
            end = no;
        }

        for(uint32_t j = 0; j < ncov; ++j){
            for(uint32_t i = start; i < end; ++i){
                result[i] += cov[j*no+i] * v[j + ni]; 
            }
        }
    }
    return;

}


// [[Rcpp::export]]
SEXP NewDense(List pgen, IntegerVector variant_subset, Nullable<NumericVector> meanimp=R_NilValue, Nullable<NumericMatrix> covariates=R_NilValue) {
   if (strcmp_r_c(pgen[0], "pgen")) {
    stop("pgen is not a pgen object");
  }
  XPtr<class RPgenReader> rp = as<XPtr<class RPgenReader> >(pgen[1]);
  XPtr<class dense_snp> xp(new dense_snp(), true);
  rp->LoadDense(*xp, variant_subset, covariates);
  if(meanimp.isNotNull()){
      NumericVector newmean = as<NumericVector>(meanimp);
      xp->ResetMeanImputation(newmean);
  }
  return List::create(_["class"] = "dense_snp", _["dense_snp"] = xp);
}


// [[Rcpp::export]]
NumericVector DenseTransMultv(List mat, NumericVector v) {
  if (strcmp_r_c(mat[0], "dense_snp")) {
    stop("matrix not the right type");
  }

  XPtr<class dense_snp> x = as<XPtr<class dense_snp> >(mat[1]);
  if(v.size() != x->Getnrow()){
      stop("vector size not compatible");
  }
  NumericVector result(x->Getncol());
  x->vtx(&v[0], &result[0]);
  return result;
}

// [[Rcpp::export]]
NumericVector DenseMultv(List mat, NumericVector v) {
  if (strcmp_r_c(mat[0], "dense_snp")) {
    stop("matrix not the right type");
  }
  XPtr<class dense_snp> x = as<XPtr<class dense_snp> >(mat[1]);
  if(v.size() != x->Getncol()){
      stop("vector size not compatible");
  }
  NumericVector result(x->Getnrow());
  x->xv(&v[0], &result[0]);
  return result;
}
// install.packages('/home/ruilinli/plink-ng/2.0/pgenlibr/',repo=NULL,type='source')