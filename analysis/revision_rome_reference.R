args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(normalizePath('analysis/outputs/r-library'), .libPaths()))
suppressPackageStartupMessages(library(rome))
directory <- args[1]
x <- as.matrix(read.table(file.path(directory,'X.tsv')))
y <- scan(file.path(directory,'y.tsv'),quiet=TRUE)
lambdas <- scan(file.path(directory,'lambda.tsv'),quiet=TRUE)
start <- proc.time()
fit <- rome.adaptive(x=x,y=y,lambda=lambdas,delta=.5,alpha=1,adapt=FALSE,
                     intercept=FALSE,preprocess='none',centering=FALSE,
                     screen='adaptive',KKT=TRUE,eps=1e-4,max.iter=100000)
elapsed <- proc.time()-start
write.table(t(fit$beta),file.path(directory,'rome-beta.tsv'),row.names=FALSE,col.names=FALSE,sep='\t')
write.table(fit$lambda,file.path(directory,'rome-lambda.tsv'),row.names=FALSE,col.names=FALSE)
write.table(c(elapsed[['elapsed']],sum(elapsed[c('user.self','sys.self')])),file.path(directory,'rome-time.tsv'),row.names=FALSE,col.names=FALSE)
