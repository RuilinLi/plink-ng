setClassUnion("matrixOrNULL", c("matrix", "NULL"))


setClass("PlinkMatrix", representation(ptr = "externalptr", samples = "integer", 
    variants = "integer", fname = "character", xim = "numeric", Dim = "integer", ncov = "integer", covs="matrixOrNULL"))

#' @export
PlinkMatrix <- function(fname, samples, variants, covs=NULL) {
    if (is.unsorted(samples, strictly = TRUE)) {
        stop("The subset indices must be sorted in strictly increasing order.")
    }
    
    if (length(variants) != length(unique(variants))) {
        stop("Must not have duplicated variant index.")
    }
    dimensions <- c(length(samples), length(variants))
    ncov = 0L
    if(!is.null(covs)) {
        if(nrow(covs) != dimensions[1]) {
            stop("Covariates must have same number of rows as SNP Matrix")
        }
        ncov = ncol(covs)
        dimensions[2] = dimensions[2] + ncov
    }
    
    samples <- as.integer(samples)
    variants <- as.integer(variants)

    
    new("PlinkMatrix", ptr = new("externalptr"), samples = samples, variants = variants, fname = fname, 
        Dim = dimensions, xim = 0, ncov = ncov, covs = covs
      )
}
setGeneric("actualize", function(x) standardGeneric("actualize"))

#' @export
setMethod(
    "actualize",
    "PlinkMatrix",
    function(x) {
        x@xim = double(length(x@variants))
        x@ptr = getcompactptr(x@fname, x@variants,x@samples, x@xim)
        return(x)
    }
)

#' @export
setMethod(
    "dim",
    "PlinkMatrix",
    function(x) {
        return(x@Dim)
    }
)