// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// NewDense
SEXP NewDense(List pgen, IntegerVector variant_subset, Nullable<NumericVector> meanimp);
RcppExport SEXP _pgenlibr_NewDense(SEXP pgenSEXP, SEXP variant_subsetSEXP, SEXP meanimpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type variant_subset(variant_subsetSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type meanimp(meanimpSEXP);
    rcpp_result_gen = Rcpp::wrap(NewDense(pgen, variant_subset, meanimp));
    return rcpp_result_gen;
END_RCPP
}
// DenseTransMultv
NumericVector DenseTransMultv(List mat, NumericVector v);
RcppExport SEXP _pgenlibr_DenseTransMultv(SEXP matSEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type mat(matSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(DenseTransMultv(mat, v));
    return rcpp_result_gen;
END_RCPP
}
// DenseMultv
NumericVector DenseMultv(List mat, NumericVector v);
RcppExport SEXP _pgenlibr_DenseMultv(SEXP matSEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type mat(matSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(DenseMultv(mat, v));
    return rcpp_result_gen;
END_RCPP
}
// SparseTest123
NumericMatrix SparseTest123(List mat, NumericVector y, IntegerVector group, NumericVector lambda_seq);
RcppExport SEXP _pgenlibr_SparseTest123(SEXP matSEXP, SEXP ySEXP, SEXP groupSEXP, SEXP lambda_seqSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type mat(matSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type group(groupSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lambda_seq(lambda_seqSEXP);
    rcpp_result_gen = Rcpp::wrap(SparseTest123(mat, y, group, lambda_seq));
    return rcpp_result_gen;
END_RCPP
}
// NewResponseObj
SEXP NewResponseObj(NumericVector y, String family, Nullable<NumericVector> offset);
RcppExport SEXP _pgenlibr_NewResponseObj(SEXP ySEXP, SEXP familySEXP, SEXP offsetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< String >::type family(familySEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type offset(offsetSEXP);
    rcpp_result_gen = Rcpp::wrap(NewResponseObj(y, family, offset));
    return rcpp_result_gen;
END_RCPP
}
// NewCoxResponseObj
SEXP NewCoxResponseObj(NumericVector status, IntegerVector order0, IntegerVector rankmin0, IntegerVector rankmax0, Nullable<NumericVector> offset);
RcppExport SEXP _pgenlibr_NewCoxResponseObj(SEXP statusSEXP, SEXP order0SEXP, SEXP rankmin0SEXP, SEXP rankmax0SEXP, SEXP offsetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type status(statusSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type order0(order0SEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type rankmin0(rankmin0SEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type rankmax0(rankmax0SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type offset(offsetSEXP);
    rcpp_result_gen = Rcpp::wrap(NewCoxResponseObj(status, order0, rankmin0, rankmax0, offset));
    return rcpp_result_gen;
END_RCPP
}
// NewProxObj
SEXP NewProxObj(int ni, IntegerVector group);
RcppExport SEXP _pgenlibr_NewProxObj(SEXP niSEXP, SEXP groupSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type ni(niSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type group(groupSEXP);
    rcpp_result_gen = Rcpp::wrap(NewProxObj(ni, group));
    return rcpp_result_gen;
END_RCPP
}
// FitGroupLasso
NumericMatrix FitGroupLasso(List mat, List prox, List response, NumericVector lambda_seq);
RcppExport SEXP _pgenlibr_FitGroupLasso(SEXP matSEXP, SEXP proxSEXP, SEXP responseSEXP, SEXP lambda_seqSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type mat(matSEXP);
    Rcpp::traits::input_parameter< List >::type prox(proxSEXP);
    Rcpp::traits::input_parameter< List >::type response(responseSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lambda_seq(lambda_seqSEXP);
    rcpp_result_gen = Rcpp::wrap(FitGroupLasso(mat, prox, response, lambda_seq));
    return rcpp_result_gen;
END_RCPP
}
// ComputeLambdaMax
double ComputeLambdaMax(List mat, List response, IntegerVector group);
RcppExport SEXP _pgenlibr_ComputeLambdaMax(SEXP matSEXP, SEXP responseSEXP, SEXP groupSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type mat(matSEXP);
    Rcpp::traits::input_parameter< List >::type response(responseSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type group(groupSEXP);
    rcpp_result_gen = Rcpp::wrap(ComputeLambdaMax(mat, response, group));
    return rcpp_result_gen;
END_RCPP
}
// match_sorted_snp
IntegerVector match_sorted_snp(IntegerVector chr, IntegerVector pos, IntegerVector refpos, IntegerVector refcumu);
RcppExport SEXP _pgenlibr_match_sorted_snp(SEXP chrSEXP, SEXP posSEXP, SEXP refposSEXP, SEXP refcumuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type chr(chrSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type pos(posSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type refpos(refposSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type refcumu(refcumuSEXP);
    rcpp_result_gen = Rcpp::wrap(match_sorted_snp(chr, pos, refpos, refcumu));
    return rcpp_result_gen;
END_RCPP
}
// getcompactptr
SEXP getcompactptr(String filename, IntegerVector variant_subset, Nullable<IntegerVector> sample_subset, NumericVector xim, Nullable<List> pvar, Nullable<int> raw_sample_ct);
RcppExport SEXP _pgenlibr_getcompactptr(SEXP filenameSEXP, SEXP variant_subsetSEXP, SEXP sample_subsetSEXP, SEXP ximSEXP, SEXP pvarSEXP, SEXP raw_sample_ctSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< String >::type filename(filenameSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type variant_subset(variant_subsetSEXP);
    Rcpp::traits::input_parameter< Nullable<IntegerVector> >::type sample_subset(sample_subsetSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type xim(ximSEXP);
    Rcpp::traits::input_parameter< Nullable<List> >::type pvar(pvarSEXP);
    Rcpp::traits::input_parameter< Nullable<int> >::type raw_sample_ct(raw_sample_ctSEXP);
    rcpp_result_gen = Rcpp::wrap(getcompactptr(filename, variant_subset, sample_subset, xim, pvar, raw_sample_ct));
    return rcpp_result_gen;
END_RCPP
}
// getcompactptrfromPgen
SEXP getcompactptrfromPgen(List pgen, IntegerVector variant_subset, NumericVector xim, IntegerVector nsample);
RcppExport SEXP _pgenlibr_getcompactptrfromPgen(SEXP pgenSEXP, SEXP variant_subsetSEXP, SEXP ximSEXP, SEXP nsampleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type variant_subset(variant_subsetSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type xim(ximSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type nsample(nsampleSEXP);
    rcpp_result_gen = Rcpp::wrap(getcompactptrfromPgen(pgen, variant_subset, xim, nsample));
    return rcpp_result_gen;
END_RCPP
}
// NewPgen
SEXP NewPgen(String filename, Nullable<List> pvar, Nullable<int> raw_sample_ct, Nullable<IntegerVector> sample_subset);
RcppExport SEXP _pgenlibr_NewPgen(SEXP filenameSEXP, SEXP pvarSEXP, SEXP raw_sample_ctSEXP, SEXP sample_subsetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< String >::type filename(filenameSEXP);
    Rcpp::traits::input_parameter< Nullable<List> >::type pvar(pvarSEXP);
    Rcpp::traits::input_parameter< Nullable<int> >::type raw_sample_ct(raw_sample_ctSEXP);
    Rcpp::traits::input_parameter< Nullable<IntegerVector> >::type sample_subset(sample_subsetSEXP);
    rcpp_result_gen = Rcpp::wrap(NewPgen(filename, pvar, raw_sample_ct, sample_subset));
    return rcpp_result_gen;
END_RCPP
}
// GetRawSampleCt
int GetRawSampleCt(List pgen);
RcppExport SEXP _pgenlibr_GetRawSampleCt(SEXP pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(GetRawSampleCt(pgen));
    return rcpp_result_gen;
END_RCPP
}
// GetVariantCt
int GetVariantCt(List pvar_or_pgen);
RcppExport SEXP _pgenlibr_GetVariantCt(SEXP pvar_or_pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pvar_or_pgen(pvar_or_pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(GetVariantCt(pvar_or_pgen));
    return rcpp_result_gen;
END_RCPP
}
// GetAlleleCt
int GetAlleleCt(List pvar_or_pgen, int variant_num);
RcppExport SEXP _pgenlibr_GetAlleleCt(SEXP pvar_or_pgenSEXP, SEXP variant_numSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pvar_or_pgen(pvar_or_pgenSEXP);
    Rcpp::traits::input_parameter< int >::type variant_num(variant_numSEXP);
    rcpp_result_gen = Rcpp::wrap(GetAlleleCt(pvar_or_pgen, variant_num));
    return rcpp_result_gen;
END_RCPP
}
// GetMaxAlleleCt
int GetMaxAlleleCt(List pvar_or_pgen);
RcppExport SEXP _pgenlibr_GetMaxAlleleCt(SEXP pvar_or_pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pvar_or_pgen(pvar_or_pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(GetMaxAlleleCt(pvar_or_pgen));
    return rcpp_result_gen;
END_RCPP
}
// HardcallPhasePresent
bool HardcallPhasePresent(List pgen);
RcppExport SEXP _pgenlibr_HardcallPhasePresent(SEXP pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(HardcallPhasePresent(pgen));
    return rcpp_result_gen;
END_RCPP
}
// Buf
NumericVector Buf(List pgen);
RcppExport SEXP _pgenlibr_Buf(SEXP pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(Buf(pgen));
    return rcpp_result_gen;
END_RCPP
}
// AlleleCodeBuf
NumericVector AlleleCodeBuf(List pgen);
RcppExport SEXP _pgenlibr_AlleleCodeBuf(SEXP pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(AlleleCodeBuf(pgen));
    return rcpp_result_gen;
END_RCPP
}
// IntBuf
IntegerVector IntBuf(List pgen);
RcppExport SEXP _pgenlibr_IntBuf(SEXP pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(IntBuf(pgen));
    return rcpp_result_gen;
END_RCPP
}
// IntAlleleCodeBuf
IntegerVector IntAlleleCodeBuf(List pgen);
RcppExport SEXP _pgenlibr_IntAlleleCodeBuf(SEXP pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(IntAlleleCodeBuf(pgen));
    return rcpp_result_gen;
END_RCPP
}
// BoolBuf
LogicalVector BoolBuf(List pgen);
RcppExport SEXP _pgenlibr_BoolBuf(SEXP pgenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    rcpp_result_gen = Rcpp::wrap(BoolBuf(pgen));
    return rcpp_result_gen;
END_RCPP
}
// ReadHardcalls
void ReadHardcalls(List pgen, SEXP buf, int variant_num, int allele_num);
RcppExport SEXP _pgenlibr_ReadHardcalls(SEXP pgenSEXP, SEXP bufSEXP, SEXP variant_numSEXP, SEXP allele_numSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< SEXP >::type buf(bufSEXP);
    Rcpp::traits::input_parameter< int >::type variant_num(variant_numSEXP);
    Rcpp::traits::input_parameter< int >::type allele_num(allele_numSEXP);
    ReadHardcalls(pgen, buf, variant_num, allele_num);
    return R_NilValue;
END_RCPP
}
// Read
void Read(List pgen, NumericVector buf, int variant_num, int allele_num);
RcppExport SEXP _pgenlibr_Read(SEXP pgenSEXP, SEXP bufSEXP, SEXP variant_numSEXP, SEXP allele_numSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type buf(bufSEXP);
    Rcpp::traits::input_parameter< int >::type variant_num(variant_numSEXP);
    Rcpp::traits::input_parameter< int >::type allele_num(allele_numSEXP);
    Read(pgen, buf, variant_num, allele_num);
    return R_NilValue;
END_RCPP
}
// ReadAlleles
void ReadAlleles(List pgen, SEXP acbuf, int variant_num, Nullable<LogicalVector> phasepresent_buf);
RcppExport SEXP _pgenlibr_ReadAlleles(SEXP pgenSEXP, SEXP acbufSEXP, SEXP variant_numSEXP, SEXP phasepresent_bufSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< SEXP >::type acbuf(acbufSEXP);
    Rcpp::traits::input_parameter< int >::type variant_num(variant_numSEXP);
    Rcpp::traits::input_parameter< Nullable<LogicalVector> >::type phasepresent_buf(phasepresent_bufSEXP);
    ReadAlleles(pgen, acbuf, variant_num, phasepresent_buf);
    return R_NilValue;
END_RCPP
}
// ReadIntList
IntegerMatrix ReadIntList(List pgen, IntegerVector variant_subset);
RcppExport SEXP _pgenlibr_ReadIntList(SEXP pgenSEXP, SEXP variant_subsetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type variant_subset(variant_subsetSEXP);
    rcpp_result_gen = Rcpp::wrap(ReadIntList(pgen, variant_subset));
    return rcpp_result_gen;
END_RCPP
}
// ReadList
NumericMatrix ReadList(List pgen, IntegerVector variant_subset, bool meanimpute);
RcppExport SEXP _pgenlibr_ReadList(SEXP pgenSEXP, SEXP variant_subsetSEXP, SEXP meanimputeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type variant_subset(variant_subsetSEXP);
    Rcpp::traits::input_parameter< bool >::type meanimpute(meanimputeSEXP);
    rcpp_result_gen = Rcpp::wrap(ReadList(pgen, variant_subset, meanimpute));
    return rcpp_result_gen;
END_RCPP
}
// VariantScores
NumericVector VariantScores(List pgen, NumericVector weights, Nullable<IntegerVector> variant_subset);
RcppExport SEXP _pgenlibr_VariantScores(SEXP pgenSEXP, SEXP weightsSEXP, SEXP variant_subsetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< Nullable<IntegerVector> >::type variant_subset(variant_subsetSEXP);
    rcpp_result_gen = Rcpp::wrap(VariantScores(pgen, weights, variant_subset));
    return rcpp_result_gen;
END_RCPP
}
// ClosePgen
void ClosePgen(List pgen);
RcppExport SEXP _pgenlibr_ClosePgen(SEXP pgenSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    ClosePgen(pgen);
    return R_NilValue;
END_RCPP
}
// NewPvar
SEXP NewPvar(String filename);
RcppExport SEXP _pgenlibr_NewPvar(SEXP filenameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< String >::type filename(filenameSEXP);
    rcpp_result_gen = Rcpp::wrap(NewPvar(filename));
    return rcpp_result_gen;
END_RCPP
}
// GetVariantId
String GetVariantId(List pvar, int variant_num);
RcppExport SEXP _pgenlibr_GetVariantId(SEXP pvarSEXP, SEXP variant_numSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pvar(pvarSEXP);
    Rcpp::traits::input_parameter< int >::type variant_num(variant_numSEXP);
    rcpp_result_gen = Rcpp::wrap(GetVariantId(pvar, variant_num));
    return rcpp_result_gen;
END_RCPP
}
// GetAlleleCode
String GetAlleleCode(List pvar, int variant_num, int allele_num);
RcppExport SEXP _pgenlibr_GetAlleleCode(SEXP pvarSEXP, SEXP variant_numSEXP, SEXP allele_numSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pvar(pvarSEXP);
    Rcpp::traits::input_parameter< int >::type variant_num(variant_numSEXP);
    Rcpp::traits::input_parameter< int >::type allele_num(allele_numSEXP);
    rcpp_result_gen = Rcpp::wrap(GetAlleleCode(pvar, variant_num, allele_num));
    return rcpp_result_gen;
END_RCPP
}
// ClosePvar
void ClosePvar(List pvar);
RcppExport SEXP _pgenlibr_ClosePvar(SEXP pvarSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pvar(pvarSEXP);
    ClosePvar(pvar);
    return R_NilValue;
END_RCPP
}
// NewSparse
SEXP NewSparse(List pgen, IntegerVector variant_subset);
RcppExport SEXP _pgenlibr_NewSparse(SEXP pgenSEXP, SEXP variant_subsetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type pgen(pgenSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type variant_subset(variant_subsetSEXP);
    rcpp_result_gen = Rcpp::wrap(NewSparse(pgen, variant_subset));
    return rcpp_result_gen;
END_RCPP
}
// SparseTransMultv
NumericVector SparseTransMultv(List mat, NumericVector v);
RcppExport SEXP _pgenlibr_SparseTransMultv(SEXP matSEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type mat(matSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(SparseTransMultv(mat, v));
    return rcpp_result_gen;
END_RCPP
}
// SparseMultv
NumericVector SparseMultv(List mat, NumericVector v);
RcppExport SEXP _pgenlibr_SparseMultv(SEXP matSEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type mat(matSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(SparseMultv(mat, v));
    return rcpp_result_gen;
END_RCPP
}
// GetSparseMeanImputation
NumericVector GetSparseMeanImputation(List mat);
RcppExport SEXP _pgenlibr_GetSparseMeanImputation(SEXP matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type mat(matSEXP);
    rcpp_result_gen = Rcpp::wrap(GetSparseMeanImputation(mat));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_pgenlibr_NewDense", (DL_FUNC) &_pgenlibr_NewDense, 3},
    {"_pgenlibr_DenseTransMultv", (DL_FUNC) &_pgenlibr_DenseTransMultv, 2},
    {"_pgenlibr_DenseMultv", (DL_FUNC) &_pgenlibr_DenseMultv, 2},
    {"_pgenlibr_SparseTest123", (DL_FUNC) &_pgenlibr_SparseTest123, 4},
    {"_pgenlibr_NewResponseObj", (DL_FUNC) &_pgenlibr_NewResponseObj, 3},
    {"_pgenlibr_NewCoxResponseObj", (DL_FUNC) &_pgenlibr_NewCoxResponseObj, 5},
    {"_pgenlibr_NewProxObj", (DL_FUNC) &_pgenlibr_NewProxObj, 2},
    {"_pgenlibr_FitGroupLasso", (DL_FUNC) &_pgenlibr_FitGroupLasso, 4},
    {"_pgenlibr_ComputeLambdaMax", (DL_FUNC) &_pgenlibr_ComputeLambdaMax, 3},
    {"_pgenlibr_match_sorted_snp", (DL_FUNC) &_pgenlibr_match_sorted_snp, 4},
    {"_pgenlibr_getcompactptr", (DL_FUNC) &_pgenlibr_getcompactptr, 6},
    {"_pgenlibr_getcompactptrfromPgen", (DL_FUNC) &_pgenlibr_getcompactptrfromPgen, 4},
    {"_pgenlibr_NewPgen", (DL_FUNC) &_pgenlibr_NewPgen, 4},
    {"_pgenlibr_GetRawSampleCt", (DL_FUNC) &_pgenlibr_GetRawSampleCt, 1},
    {"_pgenlibr_GetVariantCt", (DL_FUNC) &_pgenlibr_GetVariantCt, 1},
    {"_pgenlibr_GetAlleleCt", (DL_FUNC) &_pgenlibr_GetAlleleCt, 2},
    {"_pgenlibr_GetMaxAlleleCt", (DL_FUNC) &_pgenlibr_GetMaxAlleleCt, 1},
    {"_pgenlibr_HardcallPhasePresent", (DL_FUNC) &_pgenlibr_HardcallPhasePresent, 1},
    {"_pgenlibr_Buf", (DL_FUNC) &_pgenlibr_Buf, 1},
    {"_pgenlibr_AlleleCodeBuf", (DL_FUNC) &_pgenlibr_AlleleCodeBuf, 1},
    {"_pgenlibr_IntBuf", (DL_FUNC) &_pgenlibr_IntBuf, 1},
    {"_pgenlibr_IntAlleleCodeBuf", (DL_FUNC) &_pgenlibr_IntAlleleCodeBuf, 1},
    {"_pgenlibr_BoolBuf", (DL_FUNC) &_pgenlibr_BoolBuf, 1},
    {"_pgenlibr_ReadHardcalls", (DL_FUNC) &_pgenlibr_ReadHardcalls, 4},
    {"_pgenlibr_Read", (DL_FUNC) &_pgenlibr_Read, 4},
    {"_pgenlibr_ReadAlleles", (DL_FUNC) &_pgenlibr_ReadAlleles, 4},
    {"_pgenlibr_ReadIntList", (DL_FUNC) &_pgenlibr_ReadIntList, 2},
    {"_pgenlibr_ReadList", (DL_FUNC) &_pgenlibr_ReadList, 3},
    {"_pgenlibr_VariantScores", (DL_FUNC) &_pgenlibr_VariantScores, 3},
    {"_pgenlibr_ClosePgen", (DL_FUNC) &_pgenlibr_ClosePgen, 1},
    {"_pgenlibr_NewPvar", (DL_FUNC) &_pgenlibr_NewPvar, 1},
    {"_pgenlibr_GetVariantId", (DL_FUNC) &_pgenlibr_GetVariantId, 2},
    {"_pgenlibr_GetAlleleCode", (DL_FUNC) &_pgenlibr_GetAlleleCode, 3},
    {"_pgenlibr_ClosePvar", (DL_FUNC) &_pgenlibr_ClosePvar, 1},
    {"_pgenlibr_NewSparse", (DL_FUNC) &_pgenlibr_NewSparse, 2},
    {"_pgenlibr_SparseTransMultv", (DL_FUNC) &_pgenlibr_SparseTransMultv, 2},
    {"_pgenlibr_SparseMultv", (DL_FUNC) &_pgenlibr_SparseMultv, 2},
    {"_pgenlibr_GetSparseMeanImputation", (DL_FUNC) &_pgenlibr_GetSparseMeanImputation, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_pgenlibr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
