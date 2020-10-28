Rcpp::compileAttributes("/home/ruilinli/dev/plink-ng/2.0/pgenlibr")
install.packages("/home/ruilinli/dev/plink-ng/2.0/pgenlibr", repo=NULL,type='source')

library(pgenlibr)


pgen <- pgenlibr::NewPgen("/local-scratch/mrivas/ukb24983_exomeOQFE.pgen", pvar = NULL, 
                          sample_subset =1:1000)
m = ReadList(pgen, 1:10, meanimpute =TRUE)
r = rep(1.0,1000)
for(i in 1:10){
    print(sum(r * m[,i]))
}

m2 <- testing(pgen, 1:10)
mu = 0.0
sigma = 1.0
test_multiply(m2, r, mu, sigma)





# GetMaxAlleleCt(pgen)
# m = ReadList(pgen, 1:10)
# m = m[, 1]
# m[is.na(m)] = 3
# sum(m!=0)
# m[8]
# m[54]
# m[106]
# m[110]
# m[181]
# m[199]

# testing(pgen, 1:10)