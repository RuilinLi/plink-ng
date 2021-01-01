#ifndef __PGENLIBR_H__
#define __PGENLIBR_H__

#include "pgenlib_ffi_support.h"
#include "include/pgenlib_read.h"
#include "pvar.h"  // includes Rcpp

typedef struct CSB_ind{
   uint32_t row;
   uint32_t col;
} CSB_ind;



// A compressed sparse block format for sparse genetic matrices
class sparse_snp{
   private:
      uint64_t *blk_ptr; // the number of non-zero entries could be quite large
      uint32_t rowblock; // number of row blocks
      uint32_t colblock; // number of column blocks
      uint32_t no; // number of rows
      uint32_t ni; // number of columns
      uint32_t ndense; // number of dense columns
      CSB_ind * index; // row and column indices
      // stores the value of the non-zero entries
      // Each block should have a multiple of 256 entries so that
      // the diffvec of each block can be 64-byte aligned
      // pad with zeros if not.
      uintptr_t *diffvec;

      // This is for snps that do not use the sparse representation 
      uintptr_t *genovec;
      // Column index of the snps that do no use sparse representation
      uint32_t *dense_ind;
      uint32_t genovec_word_ct; // save this since easy to make mistake here

      bool loaded;
      double *xim; //imputed mean
      friend class RPgenReader;
   
   public:
      sparse_snp();
      ~sparse_snp();
      // Compute result = result + step * v^T X
      void vtx(const double *v, double *result, double step) const;
      // Compute result = X v
      void xv(const double *v, double *result) const;
      uint32_t Getnrow() const;
      uint32_t Getncol() const;
};


class RPgenReader {
public:
  // imitates Python/pgenlib.pyx
  RPgenReader();

#if __cplusplus >= 201103L
  RPgenReader(const RPgenReader&) = delete;
  RPgenReader& operator=(const RPgenReader&) = delete;
#endif

  void Load(String filename, Nullable<List> pvar, Nullable<int> raw_sample_ct,
            Nullable<IntegerVector> sample_subset_1based);

  uint32_t GetRawSampleCt() const;

  uint32_t GetSubsetSize() const;

  uint32_t GetVariantCt() const;

  uint32_t GetAlleleCt(uint32_t variant_idx) const;

  uint32_t GetMaxAlleleCt() const;

  bool HardcallPhasePresent() const;

  void ReadIntHardcalls(IntegerVector buf, int variant_idx, int allele_idx);

  void ReadHardcalls(NumericVector buf, int variant_idx, int allele_idx);

  void Read(NumericVector buf, int variant_idx, int allele_idx);

  void ReadAlleles(IntegerMatrix acbuf,
                   Nullable<LogicalVector> phasepresent_buf, int variant_idx);

  void ReadAllelesNumeric(NumericMatrix acbuf,
                          Nullable<LogicalVector> phasepresent_buf,
                          int variant_idx);

  void ReadIntList(IntegerMatrix buf, IntegerVector variant_subset);

  void ReadList(NumericMatrix buf, IntegerVector variant_subset, bool meanimpute);

  void FillVariantScores(NumericVector result, NumericVector weights, Nullable<IntegerVector> variant_subset);

  void ReadDiffListOrGenovec(IntegerVector variant_subset);
  void ReadCompactListNoDosage(uintptr_t** Mptr , IntegerVector variant_subset, double *xm);
  void LoadSparse(sparse_snp &x, const int *variant_subset, const uint32_t vsubset_size);

  void Close();

  ~RPgenReader();

private:
  plink2::PgenFileInfo* _info_ptr;
  plink2::RefcountedWptr* _allele_idx_offsetsp;
  plink2::RefcountedWptr* _nonref_flagsp;
  plink2::PgenReader* _state_ptr;
  uintptr_t* _subset_include_vec;
  uintptr_t* _subset_include_interleaved_vec;
  uint32_t* _subset_cumulative_popcounts;
  plink2::PgrSampleSubsetIndex _subset_index;
  uint32_t _subset_size;

  plink2::PgenVariant _pgv;

  plink2::VecW* _transpose_batch_buf;
  // kPglNypTransposeBatch (= 256) variants at a time, and then transpose
  uintptr_t* _multivar_vmaj_geno_buf;
  uintptr_t* _multivar_vmaj_phasepresent_buf;
  uintptr_t* _multivar_vmaj_phaseinfo_buf;
  uintptr_t* _multivar_smaj_geno_batch_buf;
  uintptr_t* _multivar_smaj_phaseinfo_batch_buf;
  uintptr_t* _multivar_smaj_phasepresent_batch_buf;

  void SetSampleSubsetInternal(IntegerVector sample_subset_1based);

  void ReadAllelesPhasedInternal(int variant_idx);
};

#endif