from pgenlib import PgenReader, SnpMatrix
import numpy as np

# Define the subsets of samples and variables to load in memory
# The subsamples index must be increasing
n = 10000
p = 100
subsample = np.array(range(n), np.uint32)
variant_idxs = np.array(range(p), np.uint32)


# A pgenReader object needs to be loaded first
pr = PgenReader(b'/local-scratch/mrivas/ukb24983_exomeOQFE.pgen', sample_subset=subsample)

# These two methods gets the raw number of samples and the number of variables
pr.get_raw_sample_ct()
pr.get_variant_ct()

# Load compact matrix in memory using the pgenReader object and the variant index
X = SnpMatrix()
X.load_matrix(pr, variant_idxs)


# You can run left and right matrix vector multiplication with X
# For now only works when the vector is float32
r = np.random.normal(size=n).astype(np.float32)
v = np.random.normal(size=p).astype(np.float32)
test = X.right_mulv(r)  #r^T X
test2 = X.left_mulv(v)  # X * v



# Compare with result when we load the matrix to float
ref_matrixi = np.empty([p, n], np.int8)
pr.read_list(variant_idxs, ref_matrixi)
ref_matrix = np.empty([p, n], np.float32)

#  mean imputation
for i in range(p):
    na_ind = (ref_matrixi[i, :] == -9)
    xmean = sum( ref_matrixi[i, :]* (1-na_ind))/(n - sum(na_ind))
    ref_matrix[i,:] = ref_matrixi[i, :].astype(np.float32)
    ref_matrix[i,na_ind] = xmean

ref = np.matmul(ref_matrix, r)
ref2 = np.matmul(v, ref_matrix)

print((ref - test))
print(sum(abs(ref2 - test2)))
