setClass("PlinkMatrix", representation(ptr = "externalptr", samples = "integer", 
    variants = "integer", fname = "character", xim = "numeric", xm = "numeric", xs = "numeric"), contains = "matrix")

#' @export
PlinkMatrix <- function(fname, samples, variants) {
    if (is.unsorted(samples, strictly = TRUE)) {
        stop("The subset indices must be sorted in strictly increasing order.")
    }
    
    if (length(variants) != length(unique(variants))) {
        stop("Must not have duplicated variant index.")
    }
    
    samples <- as.integer(samples)
    variants <- as.integer(variants)

    
    new("PlinkMatrix", ptr = new("externalptr"), samples = samples, variants = variants, fname = fname, 
        Dim = c(length(samples), length(variants)), xim = 0, xm = 0, xs = 0
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