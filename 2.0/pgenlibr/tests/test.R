library(pgenlibr)
library(myglmnet)
library(glmnet)
genofile = "/oak/stanford/groups/mrivas/ukbb24983/array-combined/pgen/ukb24983_cal_hla_cnv.pgen"
#genofile = "/local-scratch/mrivas/pgen/ukb24983_cal_hla_cnv.pgen"
n = 3000
p = 1000


# 1. Test loading data speed

t1 = Sys.time()
pgen <- pgenlibr::NewPgen(genofile, pvar = NULL, sample_subset =1:n)
m2 = ReadList(pgen, 1:p, meanimpute =T)
print(dim(m2))
print(Sys.time() - t1)


xm = apply(m2, 2, mean)
var_ind = which(xm>0.005)
m2 = ReadList(pgen, var_ind, meanimpute =T)


t1 = Sys.time()
m = PlinkMatrix(genofile, 1:n, var_ind)
m = actualize(m)
print(dim(m))
print(Sys.time() - t1)

p = length(var_ind)


# 2. Simulate some data to test glmnet
set.seed(1)
beta = rep(0.0, p)
d = min(p, 500)
beta[1:d] = rnorm(d)
y =  m2 %*% beta
y = y  + 0.1
print(sum(is.na(y)))
print("start fitting")
t1 = Sys.time()
a1 = myglmnet(m, y, "gaussian", standardize=F, intercept=T, thresh=1e-7)
print(a1$npasses)
print(Sys.time() - t1)


# all_lam = a1$lambda[40]

# for(i in 1:length(all_lam)){
#     lam = all_lam[i]
#     a3 = glmnet(m2, y, "gaussian", standardize=F, intercept=F, lambda=lam)
#     a2 = myglmnet(m2, y, "gaussian", standardize=F, intercept=F, lambda=lam)
#     print(sprintf("Iteration %d, glmnet npass is %d, myglmnet npass is %d", i, a3$npasses, a2$npasses))
# }

t1 = Sys.time()
a3 = glmnet(m2, y, "gaussian", standardize=F, intercept=T)
print(a3$npasses)
print(Sys.time() - t1)

t1 = Sys.time()
a2 = myglmnet(m2, y, "gaussian", standardize=F, intercept=T, thresh=1e-7)
print(a2$npasses)
print(Sys.time() - t1)

# testing warm start
beta0 = a1$beta[,50]
lam = a1$lambda[51:length(a1$lambda)]
a11 = myglmnet(m, y, "gaussian", standardize=F, intercept=T, lambda = lam, thresh=1e-7, beta0=beta0)


y2 = rep(1.0, n)
y2[y<median(y)] = 0.0
t1 = Sys.time()
a1 = myglmnet(m, y2, "logistic", standardize=F, intercept=T, lambda.min.ratio=0.5 , nlambda = 3, thresh=1e-7)
print(a1$npasses)
print(Sys.time() - t1)

t1 = Sys.time()
a3 = glmnet(m2, y2, "binomial", standardize=F, lambda.min.ratio=0.5 , nlambda = 3,intercept=T)
print(a3$npasses)
print(Sys.time() - t1)



chr = seq(5)
pos = as.integer(c(3,1,5,2,4))
refpos = rep(seq(10),5)
refcumu = (0:5)*10
match_sorted_snp(chr, pos, refpos, refcumu)